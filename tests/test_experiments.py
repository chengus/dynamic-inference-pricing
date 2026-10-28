import pytest

from auction_sim.__main__ import main
from auction_sim.experiments.runner import _candidate_dicts
from auction_sim.experiments.sweeps import build_sweep_configs, run_sweep
from auction_sim.demand import DemandConfig, make_workload


def _sweep_config():
    prices = [0.004] * 24
    return {
        "name": "base",
        "horizon_s": 60,
        "demand": {"rate_rps": 2.0, "agent_rate_rps": .3},
        "economics": {"marginal_cost_per_token": .001},
        "candidate_grid": {
            "uniform_auction": {"reserve_per_token": [.001]},
            "first_price_hybrid": {"reserve_per_token": [.001]},
            "flat_price": {"price_per_token": [.004], "reserve_per_token": [.001]},
            "static_tiers": {"base_price_per_token": [.008], "reserve_per_token": [.001],
                             "standard_wait_s": [30], "flex_wait_s": [300]},
            "time_of_day_grid": {"prices_per_token": [prices], "reserve_per_token": [.001]},
            "utilization_controller": {"initial_price_per_token": [.006], "reserve_per_token": [.001],
                                       "target_utilization": [.8], "update_interval_s": [60], "gain": [.1]},
        },
    }


def test_grid_expansion_is_cartesian_and_ordered():
    assert _candidate_dicts({"a": [1, 2], "b": ["x", "y"]}) == [
        {"a": 1, "b": "x"}, {"a": 1, "b": "y"},
        {"a": 2, "b": "x"}, {"a": 2, "b": "y"}]


def test_simulation_cli_requires_explicit_opt_in():
    with pytest.raises(SystemExit):
        main(["run", "--config", "does-not-need-to-exist.yaml"])


def test_calibration_cli_requires_explicit_opt_in():
    with pytest.raises(SystemExit):
        main(["calibrate", "--config", "does-not-need-to-exist.yaml"])


def test_sweep_cli_requires_explicit_opt_in():
    with pytest.raises(SystemExit):
        main(["sweep", "--config", "does-not-need-to-exist.yaml",
              "--family", "load", "--values", "low,high"])


def test_sweep_builders_deep_copy_and_cover_load_reserve_and_slack():
    base = _sweep_config()
    loads = build_sweep_configs(base, "load", ["low", "high"])
    reserves = build_sweep_configs(base, "reserve", [.002, .004])
    slacks = build_sweep_configs(base, "reservation_slack", [1, 1.5])
    assert [x["demand"]["rate_rps"] for x in loads] == [.5, 6.0]
    assert [x["candidate_grid"]["uniform_auction"]["reserve_per_token"] for x in reserves] == [[.002], [.004]]
    assert [x["demand"]["reservation_slack"] for x in slacks] == [1.0, 1.5]
    assert "reservation_slack" not in base["demand"]
    assert base["candidate_grid"]["uniform_auction"]["reserve_per_token"] == [.001]


def test_burst_sweep_preserves_expected_agent_overlay_rate():
    configs = build_sweep_configs(_sweep_config(), "burstiness", [2, 8])
    idle_to_active, active_to_idle = .025, .20
    p_active = idle_to_active / (idle_to_active + active_to_idle)
    expected = [c["demand"]["agent_rate_rps"] *
                (1 + p_active * (c["demand"]["agent_rate_multiplier"] - 1)) for c in configs]
    assert expected[0] == pytest.approx(expected[1])


def test_reservation_slack_keeps_seeded_orders_matched():
    configs = build_sweep_configs(_sweep_config(), "reservation_slack", [1, 1.5])
    a = make_workload(DemandConfig(**configs[0]["demand"]), 23, 60)
    b = make_workload(DemandConfig(**configs[1]["demand"]), 23, 60)
    assert a.arrivals == b.arrivals
    assert [(j.actual_output_tokens, j.value_per_token) for j in a.jobs] == [
        (j.actual_output_tokens, j.value_per_token) for j in b.jobs]


def test_sweep_runner_requires_explicit_opt_in():
    with pytest.raises(PermissionError, match="allow_simulation"):
        run_sweep({}, "load", [.5])
