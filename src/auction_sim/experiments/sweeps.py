"""Pure experiment-sweep builders and explicitly gated sweep execution."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Iterable

from .runner import _candidate_dicts, _make_policy, run_experiment


_POLICIES = ("uniform_auction", "first_price_hybrid", "flat_price", "static_tiers",
             "time_of_day_grid", "utilization_controller")
_LOAD_PRESETS = {"low": .5, "medium": 2.0, "high": 6.0}


def _numeric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return float(value)


def build_sweep_configs(base_config: dict[str, Any], family: str,
                        values: Iterable[Any]) -> list[dict[str, Any]]:
    """Return isolated configs for a load, reserve, burstiness, or slack sweep."""
    if family not in {"load", "reserve", "burstiness", "reservation_slack"}:
        raise ValueError("family must be load, reserve, burstiness, or reservation_slack")
    values = list(values)
    if not values:
        raise ValueError("sweep values must be nonempty")
    base = deepcopy(base_config)
    configs = []
    for raw in values:
        config = deepcopy(base)
        demand = config.setdefault("demand", {})
        label = str(raw)
        if family == "load":
            load_levels = config.get("workload_levels", {})
            level_map = {**_LOAD_PRESETS, **load_levels}
            rate = _numeric(level_map[raw], f"workload_levels.{raw}") if isinstance(raw, str) and raw in level_map else _numeric(raw, "load value")
            old_rate = _numeric(demand.get("rate_rps", .5), "demand.rate_rps")
            # Preserve the relative MMPP overlay strength as the offered base load changes.
            demand["agent_rate_rps"] = float(demand.get("agent_rate_rps", 0.0)) * rate / old_rate
            demand["rate_rps"] = rate
            label = raw if isinstance(raw, str) else f"{rate:g}rps"
        elif family == "reserve":
            reserve = _numeric(raw, "reserve value")
            marginal_cost = float((config.get("economics") or {}).get("marginal_cost_per_token", 0.0))
            if reserve < marginal_cost:
                raise ValueError(f"reserve {reserve:g} is below configured marginal cost {marginal_cost:g}")
            grid = config.get("candidate_grid", {})
            if set(grid) != set(_POLICIES):
                raise ValueError("reserve sweep requires candidate grids for all six policies")
            for policy in _POLICIES:
                params = grid[policy]
                if "reserve_per_token" not in params:
                    raise ValueError(f"candidate_grid.{policy} has no reserve_per_token parameter")
                params["reserve_per_token"] = [reserve]
                try:
                    for candidate in _candidate_dicts(params):
                        _make_policy(policy, candidate)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"reserve {reserve:g} is incompatible with {policy}: {exc}") from exc
            label = f"{reserve:g}"
        elif family == "burstiness":
            multiplier = _numeric(raw, "agent_rate_multiplier")
            if multiplier < 1:
                raise ValueError("agent_rate_multiplier must be at least 1")
            base_intensity = _numeric(demand.get("agent_rate_rps", 0.0), "demand.agent_rate_rps") if demand.get("agent_rate_rps", 0.0) > 0 else 0.0
            if base_intensity <= 0:
                raise ValueError("burstiness sweep requires a positive demand.agent_rate_rps")
            idle_to_active = float(demand.get("agent_idle_to_active_s", .025))
            active_to_idle = float(demand.get("agent_active_to_idle_s", .20))
            if (not math.isfinite(idle_to_active) or not math.isfinite(active_to_idle)
                    or idle_to_active <= 0 or active_to_idle <= 0):
                raise ValueError("agent MMPP transition rates must be positive")
            active_probability = idle_to_active / (idle_to_active + active_to_idle)
            old_multiplier = float(demand.get("agent_rate_multiplier", 5.0))
            old_mean_factor = 1 + active_probability * (old_multiplier - 1)
            new_mean_factor = 1 + active_probability * (multiplier - 1)
            demand["agent_rate_rps"] = base_intensity * old_mean_factor / new_mean_factor
            demand["agent_rate_multiplier"] = multiplier
            label = f"multiplier-{multiplier:g}"
        else:
            slack = _numeric(raw, "reservation_slack")
            if slack < 1:
                raise ValueError("reservation_slack must be at least 1")
            demand["reservation_slack"] = slack
            label = f"slack-{slack:g}"
        config["name"] = f"{config.get('name', 'experiment')}-{family}-{label}"
        config["sweep"] = {"family": family, "value": raw, "label": label}
        configs.append(config)
    return configs


def run_sweep(base_config: dict[str, Any], family: str, values: Iterable[Any], *,
              allow_simulation: bool = False) -> dict[str, Any]:
    """Execute configs only after the caller explicitly opts in."""
    if not allow_simulation:
        raise PermissionError("sweeps run simulations; pass allow_simulation=True explicitly")
    configs = build_sweep_configs(base_config, family, values)
    reports = []
    for config in configs:
        report = run_experiment(config)
        report["sweep"] = config["sweep"]
        reports.append(report)
    return {"schema_version": 1, "family": family, "runs": reports}
