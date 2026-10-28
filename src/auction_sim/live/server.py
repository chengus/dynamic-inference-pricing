"""Small localhost HTTP server for a wall-clock-paced synthetic simulation."""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import urlsplit
import uuid

from ..demand import DemandConfig, make_workload
from ..engine import Simulator
from ..experiments.runner import _make_policy
from ..models import SimulationSnapshot, ServerConfig


MECHANISMS = ("uniform_auction", "first_price_hybrid")
MAX_LEVELS = 50
HISTORY_SIZE = 100
MAX_REQUEST_BYTES = 2048
_STATIC_ROOT = Path(__file__).resolve().parent.parent / "web" / "static"


class _StopRun(Exception):
    """Internal signal to abandon an obsolete worker after a restart."""


class LiveDashboard:
    """Owns one simulator worker and a bounded, public-only state view."""

    def __init__(self, config: dict[str, Any], *, seed: int = 42, speed: float = 1.0,
                 mechanism: str = "uniform_auction") -> None:
        if mechanism not in MECHANISMS:
            raise ValueError(f"mechanism must be one of {', '.join(MECHANISMS)}")
        self.config = deepcopy(config)
        self.seed = self._validate_seed(seed)
        self.speed = self._validate_speed(speed)
        self.mechanism = mechanism
        self.horizon_s = float(config.get("horizon_s", 300.0))
        if not math.isfinite(self.horizon_s) or self.horizon_s < 0:
            raise ValueError("horizon_s must be finite and nonnegative")
        self.name = str(config.get("name", "synthetic live simulation"))
        grid = config.get("candidate_grid", {}).get(mechanism, {})
        configured_reserve = grid.get("reserve_per_token", [
            (config.get("economics") or {}).get("marginal_cost_per_token", 0.001)
        ])
        self.reserve_per_token = float(configured_reserve[0])
        self._condition = threading.Condition()
        self._lifecycle_lock = threading.RLock()
        self._sequence = 0
        self._generation = 0
        self._run_id = ""
        self._status = "running"
        self._error: str | None = None
        self._paused = False
        self._paused_since: float | None = None
        self._snapshot: SimulationSnapshot | None = None
        self._price_history: deque[dict[str, Any]] = deque(maxlen=HISTORY_SIZE)
        self._activity: deque[dict[str, Any]] = deque(maxlen=HISTORY_SIZE)
        self._last_price_key: tuple[float, float, int] | None = None
        self._last_admitted_count = 0
        self._last_terminal_counts = (0, 0, 0)
        self._worker: threading.Thread | None = None
        self._start_worker_locked()

    @staticmethod
    def _validate_seed(seed: Any) -> int:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        return seed

    @staticmethod
    def _validate_speed(speed: Any) -> float:
        if isinstance(speed, bool) or not isinstance(speed, (int, float)):
            raise ValueError("speed must be a number")
        speed = float(speed)
        if not math.isfinite(speed) or not 0.05 <= speed <= 100:
            raise ValueError("speed must be between 0.05 and 100")
        return speed

    def _policy(self):
        grid = self.config.get("candidate_grid", {})
        params = grid.get(self.mechanism, {})
        selected = {key: values[0] for key, values in params.items() if isinstance(values, list) and values}
        if "reserve_per_token" not in selected:
            selected["reserve_per_token"] = float(
                (self.config.get("economics") or {}).get("marginal_cost_per_token", 0.001)
            )
        return _make_policy(self.mechanism, selected)

    def _start_worker_locked(self) -> None:
        self._generation += 1
        generation = self._generation
        self._run_id = uuid.uuid4().hex[:12]
        self._sequence = 0
        self._status = "running"
        self._error = None
        self._paused = False
        self._paused_since = None
        self._snapshot = None
        self._price_history.clear()
        self._activity.clear()
        self._last_price_key = None
        self._last_admitted_count = 0
        self._last_terminal_counts = (0, 0, 0)
        self._worker = threading.Thread(
            target=self._run, args=(generation,), name=f"auction-live-{self._run_id}", daemon=True
        )
        self._worker.start()

    def _run(self, generation: int) -> None:
        try:
            server = ServerConfig(**(self.config.get("server") or {}))
            demand = DemandConfig(**(self.config.get("demand") or {}))
            workload = make_workload(demand, self.seed, self.horizon_s)
            Simulator(server).run(
                workload.jobs, self._policy(), workload.horizon_s,
                observer=lambda snapshot: self._on_snapshot(generation, snapshot),
            )
            with self._condition:
                if generation == self._generation:
                    self._status = "finished"
                    self._paused = False
                    self._sequence += 1
                    self._condition.notify_all()
        except _StopRun:
            return
        except Exception as exc:  # surface worker failures to the local dashboard
            with self._condition:
                if generation == self._generation:
                    self._status = "error"
                    self._error = f"{type(exc).__name__}: {exc}"[:300]
                    self._paused = False
                    self._sequence += 1
                    self._condition.notify_all()

    def _on_snapshot(self, generation: int, snapshot: SimulationSnapshot) -> None:
        with self._condition:
            self._ensure_generation(generation)
            previous_time = self._snapshot.time_s if self._snapshot is not None else snapshot.time_s
            remaining_virtual = max(0.0, snapshot.time_s - previous_time)
            while remaining_virtual > 0:
                self._ensure_generation(generation)
                if self._paused:
                    self._condition.wait()
                    continue
                speed = self.speed
                started = time.monotonic()
                self._condition.wait(timeout=remaining_virtual / speed)
                ended = time.monotonic()
                elapsed = ended - started
                if self._paused:
                    pause_at = self._paused_since if self._paused_since is not None else ended
                    elapsed = min(elapsed, max(0.0, pause_at - started))
                remaining_virtual = max(0.0, remaining_virtual - elapsed * speed)
            self._ensure_generation(generation)
            while self._paused:
                self._condition.wait()
                self._ensure_generation(generation)
            self._snapshot = snapshot
            self._sequence += 1
            self._record_history(snapshot)
            self._condition.notify_all()

    def _ensure_generation(self, generation: int) -> None:
        if generation != self._generation:
            raise _StopRun()

    def _record_history(self, snapshot: SimulationSnapshot) -> None:
        point = snapshot.latest_clearing_price
        if point is not None:
            key = (point.time_s, point.price_per_token, point.admissions)
            if key != self._last_price_key:
                self._last_price_key = key
                self._price_history.append({
                    "time_s": point.time_s,
                    "price_per_token": point.price_per_token,
                    "admissions": point.admissions,
                })
        if "admission" in snapshot.event:
            difference = snapshot.admitted_count - self._last_admitted_count
            if difference > 0:
                self._activity.append({"time_s": snapshot.time_s, "kind": "admission", "count": difference})
        self._last_admitted_count = snapshot.admitted_count
        counts = (snapshot.completed_count, snapshot.timeout_count, snapshot.infeasible_count)
        labels = ("completed", "timeout", "infeasible")
        for old, new, label in zip(self._last_terminal_counts, counts, labels):
            if new > old:
                self._activity.append({"time_s": snapshot.time_s, "kind": label, "count": new - old})
        self._last_terminal_counts = counts
        if snapshot.event in {"arrival", "release", "expiry"} or "+" in snapshot.event:
            self._activity.append({"time_s": snapshot.time_s, "kind": snapshot.event, "count": len(snapshot.pending)})

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            current = self._snapshot
            if current is None:
                pending: tuple = ()
                active: tuple = ()
                free = None
                completed = timeout = infeasible = admitted = 0
                revenue = 0.0
                point = None
                sim_time = 0.0
            else:
                pending, active, free = current.pending, current.active, current.free_capacity
                completed, timeout, infeasible = current.completed_count, current.timeout_count, current.infeasible_count
                admitted, revenue = current.admitted_count, current.revenue_observed
                point, sim_time = current.latest_clearing_price, current.time_s
            input_book, output_book = _build_books(pending)
            state: dict[str, Any] = {
                "status": self._status,
                "sequence": self._sequence,
                "run_id": self._run_id,
                "sim_time_s": sim_time,
                "horizon_s": self.horizon_s,
                "speed": self.speed,
                "mechanism": self.mechanism,
                "seed": self.seed,
                "label": "Synthetic simulation",
                "name": self.name,
                "load_profile": self.name,
                "busy": current.busy if current else False,
                "reserved_kv_tokens": sum(item.reserved_kv_tokens for item in active),
                "max_kv_tokens": free.max_kv_tokens if free else ServerConfig(**(self.config.get("server") or {})).max_kv_tokens,
                "pending_count": len(pending),
                "active_count": len(active),
                "completed_count": completed,
                "timeout_count": timeout,
                "infeasible_count": infeasible,
                "admitted_count": admitted,
                "revenue_observed": revenue,
                "latest_clearing_price": point.price_per_token if point else None,
                "reserve_per_token": self.reserve_per_token,
                "total_pending_input_tokens": input_book["total_input_tokens"],
                "total_pending_output_tokens": input_book["total_output_tokens"],
                "error": self._error,
                "price_history": list(self._price_history),
                "activity": list(self._activity),
                "books": {"input": input_book, "output": output_book},
            }
            if free is not None:
                state["free_batch_slots"] = free.batch_slots
                state["max_batch_size"] = free.max_batch_size
                state["batch_capacity"] = free.max_batch_size
                state["free_kv_tokens"] = free.kv_tokens
            server_config = ServerConfig(**(self.config.get("server") or {}))
            state.setdefault("max_batch_size", server_config.max_batch_size)
            state.setdefault("batch_capacity", server_config.max_batch_size)
            state.setdefault("free_batch_slots", server_config.max_batch_size)
            state.setdefault("free_kv_tokens", server_config.max_kv_tokens)
            return state

    def control(self, body: dict[str, Any]) -> dict[str, Any]:
        with self._lifecycle_lock:
            return self._control(body)

    def _control(self, body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise ValueError("control body must be a JSON object")
        action = body.get("action")
        if action == "pause":
            with self._condition:
                if self._status == "running":
                    self._paused = True
                    self._paused_since = time.monotonic()
                    self._status = "paused"
                    self._sequence += 1
                    self._condition.notify_all()
        elif action == "resume":
            with self._condition:
                if self._status == "paused":
                    self._paused = False
                    self._paused_since = None
                    self._status = "running"
                    self._sequence += 1
                    self._condition.notify_all()
        elif action == "speed":
            value = self._validate_speed(body.get("speed"))
            with self._condition:
                self.speed = value
                self._sequence += 1
                self._condition.notify_all()
        elif action == "restart":
            self._restart(seed=self.seed, mechanism=self.mechanism)
        elif action == "mechanism":
            mechanism = body.get("mechanism")
            if mechanism not in MECHANISMS:
                raise ValueError(f"mechanism must be one of {', '.join(MECHANISMS)}")
            self._restart(seed=self.seed, mechanism=mechanism)
        else:
            raise ValueError("action must be pause, resume, restart, speed, or mechanism")
        return self.snapshot()

    def _restart(self, *, seed: int, mechanism: str) -> None:
        with self._lifecycle_lock:
            with self._condition:
                old_worker = self._worker
                self._generation += 1
                self.mechanism = mechanism
                self.seed = self._validate_seed(seed)
                self._paused = False
                self._paused_since = None
                grid = self.config.get("candidate_grid", {}).get(mechanism, {})
                configured_reserve = grid.get("reserve_per_token", [
                    (self.config.get("economics") or {}).get("marginal_cost_per_token", 0.001)
                ])
                self.reserve_per_token = float(configured_reserve[0])
                self._condition.notify_all()
            if old_worker is not None and old_worker is not threading.current_thread():
                old_worker.join(timeout=2.0)
            with self._condition:
                self._start_worker_locked()

    def close(self) -> None:
        """Stop the worker and wake any event-pacing wait before returning."""
        with self._lifecycle_lock:
            with self._condition:
                worker = self._worker
                self._generation += 1
                self._paused = False
                self._paused_since = None
                self._condition.notify_all()
            if worker is not None and worker is not threading.current_thread():
                worker.join(timeout=2.0)


def _build_books(pending) -> tuple[dict[str, Any], dict[str, Any]]:
    grouped: dict[float, dict[str, int]] = defaultdict(lambda: {"orders": 0, "input_tokens": 0, "output_tokens": 0})
    for order in pending:
        # Floor to a stable 0.001/token bucket; tiny epsilon avoids binary-float
        # representation shifting an exact boundary down one level.
        level = math.floor(order.bid_per_token / 0.001 + 1e-12) * 0.001
        row = grouped[level]
        row["orders"] += 1
        row["input_tokens"] += order.prompt_tokens
        row["output_tokens"] += order.max_tokens
    all_rows = []
    cumulative_orders = cumulative_input = cumulative_output = 0
    for price in sorted(grouped, reverse=True):
        row = grouped[price]
        cumulative_orders += row["orders"]
        cumulative_input += row["input_tokens"]
        cumulative_output += row["output_tokens"]
        all_rows.append({
            "bid_per_token": round(price, 6),
            **row,
            "cumulative_orders": cumulative_orders,
            "cumulative_input_tokens": cumulative_input,
            "cumulative_output_tokens": cumulative_output,
        })
    total_orders = len(pending)
    total_input = sum(order.prompt_tokens for order in pending)
    total_output = sum(order.max_tokens for order in pending)
    common = {"truncated": len(all_rows) > MAX_LEVELS, "total_levels": len(all_rows), "total_orders": total_orders,
              "total_input_tokens": total_input, "total_output_tokens": total_output}
    return ({"levels": all_rows[:MAX_LEVELS], **common}, {"levels": all_rows[:MAX_LEVELS], **common})


def make_server(config: dict[str, Any], *, host: str = "127.0.0.1", port: int = 8765,
                seed: int = 42, speed: float = 1.0,
                mechanism: str = "uniform_auction") -> tuple[ThreadingHTTPServer, LiveDashboard]:
    """Create and start one dashboard worker; caller owns HTTP serve/close."""
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("dashboard may only bind to localhost")
    dashboard = LiveDashboard(config, seed=seed, speed=speed, mechanism=mechanism)
    handler = _handler_for(dashboard)
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except Exception:
        dashboard.close()
        raise
    server.daemon_threads = True
    server.dashboard = dashboard  # type: ignore[attr-defined]
    return server, dashboard


def serve_dashboard(config: dict[str, Any], *, host: str = "127.0.0.1", port: int = 8765,
                    seed: int = 42, speed: float = 1.0,
                    mechanism: str = "uniform_auction") -> None:
    server, dashboard = make_server(config, host=host, port=port, seed=seed,
                                    speed=speed, mechanism=mechanism)
    address, actual_port = server.server_address[:2]
    print(f"Synthetic local dashboard: http://{address}:{actual_port}/", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.close()
        server.shutdown()
        server.server_close()
        if dashboard._worker is not None:
            dashboard._worker.join(timeout=2.0)


def _handler_for(dashboard: LiveDashboard):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AuctionSimLocal/1"
        sys_version = ""

        def log_message(self, fmt: str, *args) -> None:
            return

        def _valid_origin(self) -> bool:
            host_header = self.headers.get("Host", "")
            host_name = host_header.rsplit(":", 1)[0].strip("[]").lower()
            if host_name not in {"localhost", "127.0.0.1"}:
                return False
            origin = self.headers.get("Origin")
            if not origin:
                return True
            try:
                parsed = urlsplit(origin)
                origin_host = (parsed.hostname or "").lower()
                origin_port = parsed.port or (443 if parsed.scheme == "https" else 80)
            except ValueError:
                return False
            host_port = self.server.server_address[1]
            return parsed.scheme in {"http", "https"} and origin_host in {"localhost", "127.0.0.1"} and origin_port == host_port

        def _send(self, status: int, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, status: int, value: Any) -> None:
            payload = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
            if len(payload) > 128_000:
                self._send(500, b'{"error":"response exceeds size limit"}', "application/json; charset=utf-8")
                return
            self._send(status, payload, "application/json; charset=utf-8")

        def do_GET(self) -> None:
            if not self._valid_origin():
                self._json(403, {"error": "local requests only"})
                return
            path = urlsplit(self.path).path
            if path == "/api/state":
                self._json(200, dashboard.snapshot())
                return
            relative = "index.html" if path == "/" else path.removeprefix("/static/").lstrip("/")
            candidate = (_STATIC_ROOT / relative).resolve()
            try:
                candidate.relative_to(_STATIC_ROOT.resolve())
            except ValueError:
                self._json(404, {"error": "not found"})
                return
            if not candidate.is_file():
                self._json(404, {"error": "not found"})
                return
            data = candidate.read_bytes()
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
                content_type += "; charset=utf-8"
            self._send(200, data, content_type)

        def do_POST(self) -> None:
            if not self._valid_origin():
                self._json(403, {"error": "local requests only"})
                return
            if urlsplit(self.path).path != "/api/control":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._json(400, {"error": "invalid content length"})
                return
            if length < 1 or length > MAX_REQUEST_BYTES:
                self._json(413, {"error": "request body must be 1–2048 bytes"})
                return
            try:
                body = json.loads(self.rfile.read(length))
                state = dashboard.control(body)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)[:200]})
                return
            self._json(200, state)

    return Handler
