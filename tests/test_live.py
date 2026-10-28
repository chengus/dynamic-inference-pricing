import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from auction_sim.engine import Simulator
from auction_sim.live.server import _build_books, make_server
from auction_sim.mechanisms import UniformAuction
from auction_sim.models import Job, Order, ServerConfig, SimulationSnapshot


pytestmark = pytest.mark.simulation


def _config(*, horizon=0, rate=0):
    return {
        "name": "test-live",
        "horizon_s": horizon,
        "server": {"t0_s": 0.01, "k_s": 0, "prefill_s_per_token": 0,
                   "max_batch_size": 2, "max_kv_tokens": 32},
        "demand": {"rate_rps": rate, "agent_rate_rps": 0, "output_cap_tokens": 2,
                   "reservation_slack": 1, "market_max_wait_s": 0, "max_prompt_tokens": 2},
        "candidate_grid": {
            "uniform_auction": {"reserve_per_token": [0.001]},
            "first_price_hybrid": {"reserve_per_token": [0.001]},
        },
    }


class AdmitAll:
    def step(self, pending, free_capacity, t):
        from auction_sim.models import Admission, Decision
        return Decision(tuple(Admission(order.request_id, order.bid_per_token) for order in pending[:free_capacity.batch_slots]))

    def observe(self, interval):
        pass

    def next_wakeup(self, t):
        return None


def test_observer_snapshots_are_frozen_public_and_capture_expiry():
    jobs = [
        Job(Order("done", 0, 1, 2, 4.0, 0, "interactive", "market"), 1, 99),
        Job(Order("miss", 0, 1, 2, 0.0, 0.2, "agent"), 2, 88),
    ]
    snapshots: list[SimulationSnapshot] = []
    result = Simulator(ServerConfig(t0_s=0.1, k_s=0, prefill_s_per_token=0, max_batch_size=1)).run(
        jobs, UniformAuction(1.0), 0, observer=snapshots.append
    )
    assert any("expiry" in item.event for item in snapshots)
    assert snapshots[-1].event == "final"
    assert snapshots[-1].completed_count == 1
    assert snapshots[-1].timeout_count == 1
    assert snapshots[-1].revenue_observed == pytest.approx(2)
    active = [item for snap in snapshots for item in snap.active]
    assert all(not hasattr(item, "actual_output_tokens") and not hasattr(item, "value_per_token") for item in active)
    public_orders = [order for snap in snapshots for order in snap.pending]
    assert public_orders and all(not hasattr(order, "actual_output_tokens") for order in public_orders)
    assert result.requests[0].job.actual_output_tokens == 1  # final result retains simulator truth for offline reports


def test_input_and_output_books_share_bid_levels_and_keep_full_totals():
    pending = [
        Order(f"r{i}", 0, i + 1, i + 4, 5 - i / 1000, 1, "agent")
        for i in range(55)
    ]
    input_book, output_book = _build_books(pending)
    assert input_book == output_book
    assert input_book["truncated"] is True
    assert input_book["total_orders"] == 55
    assert input_book["total_levels"] == 55
    assert input_book["total_input_tokens"] == sum(order.prompt_tokens for order in pending)
    assert input_book["total_output_tokens"] == sum(order.max_tokens for order in pending)
    assert len(input_book["levels"]) == 50
    prices = [row["bid_per_token"] for row in input_book["levels"]]
    assert prices == sorted(prices, reverse=True)


def test_observer_does_not_change_simulated_outcomes():
    jobs = [Job(Order("a", 0, 2, 3, 2, 0, "interactive", "market"), 2, 2),
            Job(Order("b", .015, 1, 2, 1, 1, "agent"), 1, 1)]
    server = ServerConfig(t0_s=.01, k_s=.001, prefill_s_per_token=.002,
                          max_batch_size=1, max_kv_tokens=8)
    observed = []
    expected = Simulator(server).run(jobs, UniformAuction(.1), .05)
    actual = Simulator(server).run(jobs, UniformAuction(.1), .05, observer=observed.append)
    assert actual == expected
    assert observed[0].event == "initial"
    assert observed[-1].event == "final"


def _http_json(server, method="GET", path="/api/state", body=None, headers=None):
    address, port = server.server_address[:2]
    payload = None if body is None else json.dumps(body).encode()
    request = Request(f"http://{address}:{port}{path}", data=payload, method=method, headers=headers or {})
    with urlopen(request, timeout=3) as response:
        return response.status, json.loads(response.read())


def test_local_http_state_controls_security_and_static_routes():
    server, dashboard = make_server(_config(), port=0)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    try:
        status, state = _http_json(server)
        assert status == 200
        assert state["status"] in {"running", "finished"}
        assert state["label"] == "Synthetic simulation"
        assert state["error"] is None
        assert state["max_kv_tokens"] == 32
        assert state["batch_capacity"] == 2
        assert state["books"]["input"]["total_levels"] >= 0
        assert "actual_output_tokens" not in json.dumps(state)
        address, port = server.server_address[:2]
        with urlopen(f"http://{address}:{port}/", timeout=3) as response:
            assert response.status == 200
            assert b"Auction" in response.read()
        for asset, content in [("styles.css", b".app-shell"), ("app.js", b"fetchState")]:
            with urlopen(f"http://{address}:{port}/static/{asset}", timeout=3) as response:
                assert response.status == 200
                assert content in response.read()
        _, paused = _http_json(server, "POST", "/api/control", {"action": "pause"})
        assert paused["status"] in {"paused", "finished"}
        _, speed = _http_json(server, "POST", "/api/control", {"action": "speed", "speed": 2})
        assert speed["speed"] == 2
        request = Request(f"http://{address}:{port}/api/control", data=b'{"action":"speed","speed":true}', method="POST",
                          headers={"Content-Type": "application/json"})
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=3)
        assert error.value.code == 400
        request = Request(f"http://{address}:{port}/api/control", data=b'{"action":"pause"}', method="POST",
                          headers={"Origin": "http://evil.example", "Content-Type": "application/json"})
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=3)
        assert error.value.code == 403
        request = Request(f"http://{address}:{port}/api/control", data=b'{"action":"pause"}', method="POST",
                          headers={"Origin": "http://[invalid", "Content-Type": "application/json"})
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=3)
        assert error.value.code == 403
        request = Request(f"http://{address}:{port}/../pyproject.toml")
        with pytest.raises(HTTPError):
            urlopen(request, timeout=3)
    finally:
        dashboard.close()
        server.shutdown()
        server.server_close()
        serving.join(timeout=2)


def test_live_worker_pause_resume_and_mechanism_restart_keep_one_run():
    server, dashboard = make_server(_config(horizon=10), port=0, speed=0.05)
    try:
        paused = dashboard.control({"action": "pause"})
        assert paused["status"] == "paused"
        frozen_at = paused["sim_time_s"]
        time.sleep(0.05)
        assert dashboard.snapshot()["sim_time_s"] == frozen_at
        dashboard.control({"action": "resume"})
        old_run = dashboard.snapshot()["run_id"]
        changed = dashboard.control({"action": "mechanism", "mechanism": "first_price_hybrid"})
        assert changed["mechanism"] == "first_price_hybrid"
        assert changed["run_id"] != old_run
        assert changed["seed"] == 42
    finally:
        dashboard.close()
        server.server_close()
