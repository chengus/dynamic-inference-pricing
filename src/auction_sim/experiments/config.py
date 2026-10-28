"""YAML configuration loading and static validation for experiment plans."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Any

import yaml

from ..demand import DemandConfig
from ..models import ServerConfig


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("experiment config must be a YAML mapping")
    validate_config(data)
    return data


def validate_config(data: dict[str, Any]) -> None:
    allowed = {"name", "horizon_s", "server", "demand", "seeds", "candidate_grid",
               "oracle", "output", "slo", "workload_levels", "economics", "sweep"}
    extra = set(data) - allowed
    if extra:
        raise ValueError(f"unknown config keys: {', '.join(sorted(extra))}")
    if not isinstance(data.get("name", "experiment"), str):
        raise ValueError("name must be a string")
    if "output" in data and (not isinstance(data["output"], str) or not data["output"]):
        raise ValueError("output must be a nonempty path string")
    if "sweep" in data:
        sweep = data["sweep"]
        valid_value = (isinstance(sweep, dict) and "value" in sweep and
                       ((isinstance(sweep["value"], str) and bool(sweep["value"])) or
                        (not isinstance(sweep["value"], bool) and isinstance(sweep["value"], (int, float))
                         and math.isfinite(sweep["value"]))))
        if (not isinstance(sweep, dict) or set(sweep) != {"family", "value", "label"}
                or not isinstance(sweep.get("family"), str)
                or sweep.get("family") not in {"load", "reserve", "burstiness", "reservation_slack"}
                or not valid_value or not isinstance(sweep.get("label"), str) or not sweep["label"]):
            raise ValueError("sweep metadata must contain a supported family, value, and nonempty label")
    levels = data.get("workload_levels", {})
    if not isinstance(levels, dict) or set(levels) - {"low", "medium", "high"}:
        raise ValueError("workload_levels may define only low, medium, and high rates")
    for name, value in levels.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"workload_levels.{name} must be a finite positive rate")
    horizon = data.get("horizon_s", 3600)
    if isinstance(horizon, bool) or not isinstance(horizon, (int, float)) or not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon_s must be positive")
    ServerConfig(**(data.get("server") or {}))
    DemandConfig(**(data.get("demand") or {}))
    seeds = data.get("seeds", {"calibration": list(range(5)), "evaluation": list(range(100, 110))})
    if not isinstance(seeds, dict) or set(seeds) != {"calibration", "evaluation"} or not seeds.get("calibration") or not seeds.get("evaluation"):
        raise ValueError("seeds must include nonempty calibration and evaluation lists")
    if any(not isinstance(seeds[key], list) for key in ("calibration", "evaluation")):
        raise ValueError("seed collections must be lists")
    all_seeds = seeds["calibration"] + seeds["evaluation"]
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in all_seeds):
        raise ValueError("seeds must be nonnegative integers")
    if len(set(seeds["calibration"])) != len(seeds["calibration"]) or len(set(seeds["evaluation"])) != len(seeds["evaluation"]):
        raise ValueError("seed lists may not contain duplicates")
    cal, test = set(seeds["calibration"]), set(seeds["evaluation"])
    if cal & test:
        raise ValueError("calibration and evaluation seeds must be disjoint")
    candidates = data.get("candidate_grid", {})
    if not isinstance(candidates, dict):
        raise ValueError("candidate_grid must be a mapping")
    known = {
        "uniform_auction": {"reserve_per_token"}, "first_price_hybrid": {"reserve_per_token"},
        "flat_price": {"price_per_token", "reserve_per_token"},
        "static_tiers": {"base_price_per_token", "reserve_per_token", "standard_wait_s", "flex_wait_s"},
        "time_of_day_grid": {"prices_per_token", "reserve_per_token", "edges_s"},
        "utilization_controller": {"initial_price_per_token", "reserve_per_token", "target_utilization",
                                   "update_interval_s", "gain", "min_price", "max_price"},
    }
    for policy, params in candidates.items():
        if policy not in known:
            raise ValueError(f"unknown policy in candidate_grid: {policy}")
        if not isinstance(params, dict) or not params or set(params) - known[policy] or any(not isinstance(v, list) or not v for v in params.values()):
            raise ValueError(f"candidate_grid.{policy} must map parameters to nonempty lists")
    economics = data.get("economics")
    if economics is not None:
        if not isinstance(economics, dict) or set(economics) != {"marginal_cost_per_token"}:
            raise ValueError("economics must contain marginal_cost_per_token")
        cost = economics["marginal_cost_per_token"]
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
            raise ValueError("economics.marginal_cost_per_token must be finite and nonnegative")
    else:
        cost = 0.0
    for policy, params in candidates.items():
        if "reserve_per_token" in params and any(value < cost for value in params["reserve_per_token"]):
            raise ValueError(f"candidate_grid.{policy}.reserve_per_token must be at least the configured marginal cost {cost:g}")
    if candidates:
        from .runner import _candidate_dicts, _make_policy
        for policy, params in candidates.items():
            for index, values in enumerate(_candidate_dicts(params)):
                try:
                    _make_policy(policy, values)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"candidate_grid.{policy} candidate {index} is invalid: {exc}") from exc
    oracle = data.get("oracle", {})
    if isinstance(oracle, bool):
        oracle = {"enabled": oracle}
    if not isinstance(oracle, dict) or set(oracle) - {"enabled", "max_jobs", "max_nodes"}:
        raise ValueError("oracle must contain only enabled, max_jobs, and max_nodes")
    if "enabled" in oracle and not isinstance(oracle["enabled"], bool):
        raise ValueError("oracle.enabled must be a boolean")
    if oracle.get("enabled", False):
        for key in ("max_jobs", "max_nodes"):
            value = oracle.get(key, 6 if key == "max_jobs" else 100000)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"oracle.{key} must be a positive integer")
    if "slo" in data:
        slo = data["slo"]
        if not isinstance(slo, dict) or set(slo) - {"fill_rate", "required_segments", "reference_rate_rps"}:
            raise ValueError("slo may contain only fill_rate, required_segments, and reference_rate_rps")
        fill = slo.get("fill_rate", .95)
        if isinstance(fill, bool) or not isinstance(fill, (int, float)) or not math.isfinite(fill) or fill <= 0 or fill > 1:
            raise ValueError("slo.fill_rate must be in (0, 1]")
        required = slo.get("required_segments", ["interactive", "agent"])
        if not isinstance(required, list) or not required or any(not isinstance(x, str) or not x for x in required) or len(set(required)) != len(required):
            raise ValueError("slo.required_segments must be a nonempty unique list of segment names")
        ref_rate = slo.get("reference_rate_rps", .05)
        if isinstance(ref_rate, bool) or not isinstance(ref_rate, (int, float)) or not math.isfinite(ref_rate) or ref_rate <= 0:
            raise ValueError("slo.reference_rate_rps must be finite and positive")


def config_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic summary without creating workloads or running the engine."""
    validate_config(data)
    return {"name": data.get("name", "experiment"), "horizon_s": data.get("horizon_s", 3600),
            "server": asdict(ServerConfig(**(data.get("server") or {}))),
            "demand": asdict(DemandConfig(**(data.get("demand") or {}))),
            "calibration_seeds": data.get("seeds", {}).get("calibration", list(range(5))),
            "evaluation_seeds": data.get("seeds", {}).get("evaluation", list(range(100, 110))),
            "candidate_policy_count": len(data.get("candidate_grid", {})),
            "candidate_grid": data.get("candidate_grid", {}),
            "economics": {"marginal_cost_per_token": (data.get("economics") or {}).get("marginal_cost_per_token", 0.0),
                          "status": "synthetic estimate" if data.get("economics") else "unspecified; no reserve cost floor"},
            "note": "Static validation only; no simulation or calibration has run."}
