"""Paired-seed calibration and frozen evaluation. Callers must opt in via the CLI."""

from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product
import math
from types import SimpleNamespace
from typing import Any

from ..demand import DemandConfig, make_workload
from ..engine import Simulator
from ..models import Admission, Decision, ServerConfig
from .metrics import summarize


class _AffordableFIFO:
    """Low-load reference policy used only to set segment latency scales."""
    def __init__(self, reserve: float):
        self.reserve = reserve
    def step(self, pending, free_capacity, t):
        admitted, kv, slots = [], free_capacity.kv_tokens, free_capacity.batch_slots
        for order in pending:
            if slots and order.bid_per_token >= self.reserve and order.reserved_kv_tokens <= kv:
                admitted.append(Admission(order.request_id, self.reserve))
                slots -= 1
                kv -= order.reserved_kv_tokens
        return Decision(tuple(admitted))
    def observe(self, interval):
        pass
    def next_wakeup(self, t):
        return None


def _candidate_dicts(grid: dict[str, list[Any]]):
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))] if keys else [{}]


def _make_policy(name: str, params: dict[str, Any]):
    from ..mechanisms import FirstPriceHybridAuction, UniformAuction
    from ..baselines import FlatPrice, StaticTiers, TimeOfDayGrid, UtilizationController
    constructors = {"uniform_auction": UniformAuction, "first_price_hybrid": FirstPriceHybridAuction,
                    "flat_price": FlatPrice, "static_tiers": StaticTiers,
                    "time_of_day_grid": TimeOfDayGrid, "utilization_controller": UtilizationController}
    aliases = {"UniformAuction": "uniform_auction", "FirstPriceHybridAuction": "first_price_hybrid",
               "FlatPrice": "flat_price", "StaticTiers": "static_tiers",
               "TimeOfDayGrid": "time_of_day_grid", "UtilizationController": "utilization_controller"}
    key = aliases.get(name, name)
    if key not in constructors:
        raise ValueError(f"unknown policy: {name}")
    return constructors[key](**params)


def _references(workloads, server: ServerConfig, demand: DemandConfig, reference_rate_rps: float,
                required_segments: tuple[str, ...]):
    if demand.rate_rps <= 0:
        raise ValueError("low-load reference requires a positive base demand rate")
    scale = reference_rate_rps / demand.rate_rps
    reference_demand = replace(demand, rate_rps=reference_rate_rps,
                               agent_rate_rps=demand.agent_rate_rps * scale)
    values: dict[str, dict[str, list[float]]] = {
        segment: {"ttft": [], "e2e": []} for segment in required_segments}
    simulator = Simulator(server)
    for workload in workloads:
        ref_workload = make_workload(reference_demand, workload.seed, workload.horizon_s)
        result = simulator.run(ref_workload.jobs, _AffordableFIFO(1e-12), ref_workload.horizon_s)
        for item in result.requests:
            segment = item.job.order.segment
            if item.status == "completed" and item.first_token_at_s is not None and item.completed_at_s is not None:
                values.setdefault(segment, {"ttft": [], "e2e": []})["ttft"].append(
                    item.first_token_at_s - item.job.order.arrival_s)
                values[segment]["e2e"].append(item.completed_at_s - item.job.order.arrival_s)
    missing = [s for s in required_segments if not values.get(s, {}).get("e2e")]
    if missing:
        raise ValueError(f"low-load reference observed no completed requests for required segments: {', '.join(missing)}")
    def p95(values):
        return values[max(0, math.ceil(.95 * len(values)) - 1)]
    return ({s: {"ttft_p95_s": p95(sorted(values[s]["ttft"])),
               "e2e_p95_s": p95(sorted(values[s]["e2e"]))} for s in required_segments},
            reference_demand)


def _attach_oracle_metrics(metrics: dict[str, Any], oracle: Any) -> None:
    """Add the shared hindsight oracle result to already-aggregated metrics."""
    oracle_welfare = getattr(oracle, "welfare", None)
    exact = bool(getattr(oracle, "exact", False))
    bound = getattr(oracle, "upper_bound", None)
    welfare = metrics["welfare"]
    exact_ratio = welfare / oracle_welfare if exact and oracle_welfare else None
    lower_ratio = welfare / bound if not exact and bound else None
    metrics["oracle"] = {
        "exact": exact, "welfare": oracle_welfare, "upper_bound": bound,
        "algorithm": getattr(oracle, "algorithm", None), "nodes": getattr(oracle, "nodes", None),
        "efficiency_ratio": exact_ratio, "certified_lower_ratio": lower_ratio,
        "efficiency_interval": ([exact_ratio, exact_ratio] if exact_ratio is not None else
                                [lower_ratio, 1.0] if lower_ratio is not None else None),
        "interpretation": "exact optimum ratio on supplied decision calendar" if exact else
        ("certified lower ratio to supplied upper bound" if bound else "no certified ratio"),
    }


def run_experiment(config: dict[str, Any], *, calibration_only: bool = False) -> dict[str, Any]:
    """Run calibration and, unless requested otherwise, frozen evaluation."""
    horizon = float(config.get("horizon_s", 3600))
    server = ServerConfig(**(config.get("server") or {}))
    demand = DemandConfig(**(config.get("demand") or {}))
    seeds = config.get("seeds", {"calibration": list(range(5)), "evaluation": list(range(100, 110))})
    cal_workloads = [make_workload(demand, s, horizon) for s in seeds["calibration"]]
    eval_workloads = [] if calibration_only else [make_workload(demand, s, horizon) for s in seeds["evaluation"]]
    grid = config.get("candidate_grid", {})
    required_policies = {"uniform_auction", "first_price_hybrid", "flat_price", "static_tiers",
                         "time_of_day_grid", "utilization_controller"}
    if set(grid) != required_policies:
        raise ValueError("run_experiment requires candidate grids for all six auction and baseline policies")
    slo_cfg = config.get("slo", {})
    fill_slo = float(slo_cfg.get("fill_rate", .95))
    required_segments = tuple(slo_cfg.get("required_segments", ["interactive", "agent"]))
    reference_rate_rps = float(slo_cfg.get("reference_rate_rps", .05))
    references, reference_demand = _references(cal_workloads, server, demand, reference_rate_rps,
                                                required_segments)
    policies: dict[str, Any] = {}
    # Aggregate each evaluation immediately. Keeping SimulationResult objects
    # for every policy/seed retains the full token-step usage trace and price
    # series, which grows quickly for multi-hour sweeps.
    eval_rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    eval_request_rows: dict[str, list[list[dict[str, Any]]]] = {}
    eval_calendars: dict[int, set[float]] = {w.seed: set() for w in eval_workloads}
    simulator = Simulator(server)
    for name, parameter_grid in grid.items():
        candidates = _candidate_dicts(parameter_grid)
        records = []
        for candidate_index, params in enumerate(candidates):
            seed_rows = []
            for workload in cal_workloads:
                result = simulator.run(workload.jobs, _make_policy(name, params), workload.horizon_s)
                seed_rows.append({"seed": workload.seed, "metrics": summarize(
                    result, server, low_load_reference=references, fill_slo=fill_slo,
                    required_segments=required_segments)})
            feasible = bool(seed_rows) and all(x["metrics"]["slo"]["all_segments_feasible"] for x in seed_rows)
            avg_revenue = sum(x["metrics"]["revenue_per_gpu_hour"] or 0 for x in seed_rows) / len(seed_rows)
            records.append({"candidate_id": candidate_index, "parameters": params, "feasible": feasible,
                            "mean_revenue_per_gpu_hour": avg_revenue, "seeds": seed_rows})
        feasible_records = [r for r in records if r["feasible"]]
        winner = max(feasible_records, key=lambda x: x["mean_revenue_per_gpu_hour"]) if feasible_records else None
        eval_rows_by_policy[name] = []
        eval_request_rows[name] = []
        if winner:
            for workload in eval_workloads:
                result = simulator.run(workload.jobs, _make_policy(name, winner["parameters"]), workload.horizon_s)
                metrics = summarize(result, server, low_load_reference=references,
                                    fill_slo=fill_slo, required_segments=required_segments)
                eval_rows_by_policy[name].append({"seed": workload.seed, "metrics": metrics})
                eval_calendars[workload.seed].update(
                    r.admitted_at_s for r in result.requests if r.admitted_at_s is not None)
                eval_request_rows[name].append([
                    {"request_id": r.job.order.request_id,
                     "segment": r.job.order.segment,
                     "order_type": r.job.order.order_type,
                     "prompt_tokens": r.job.order.prompt_tokens,
                     "actual_output_tokens": r.job.actual_output_tokens,
                     "max_tokens": r.job.order.max_tokens,
                     "status": r.status,
                     "admitted_at_s": r.admitted_at_s,
                     "first_token_at_s": r.first_token_at_s,
                     "completed_at_s": r.completed_at_s,
                     "revenue": r.revenue}
                    for r in result.requests])
        policies[name] = {"calibration_candidates": records,
                          "selected_candidate_id": winner["candidate_id"] if winner else None,
                          "selected_parameters": winner["parameters"] if winner else None,
                          "calibration_feasible_candidates": len(feasible_records),
                          "evaluation": [], "evaluation_all_seeds_feasible": False,
                          "report_status": ("calibrated_only" if winner and calibration_only else
                                            "no_calibration_feasible_candidate")}
    oracle_cfg = config.get("oracle", {})
    if isinstance(oracle_cfg, bool):
        oracle_cfg = {"enabled": oracle_cfg}
    oracle_enabled = bool(oracle_cfg.get("enabled", False)) and not calibration_only
    oracle_by_seed = {}
    if oracle_enabled:
        from ..oracle import OracleSearchLimit, exact_hindsight_oracle
        max_jobs = oracle_cfg.get("max_jobs", 6)
        max_nodes = oracle_cfg.get("max_nodes", 100000)
        for workload in eval_workloads:
            calendar = sorted(eval_calendars[workload.seed])
            if len(workload.jobs) > max_jobs:
                upper_bound = sum(job.total_value for job in workload.jobs)
                oracle_by_seed[workload.seed] = {"status": "skipped_job_limit", "job_count": len(workload.jobs),
                                                 "max_jobs": max_jobs, "decision_times_s": calendar,
                                                 "result": SimpleNamespace(welfare=None, exact=False,
                                                                            upper_bound=upper_bound,
                                                                            algorithm="trivial_offered_value_bound",
                                                                            nodes=0)}
                continue
            try:
                oracle = exact_hindsight_oracle(workload.jobs, server, workload.horizon_s,
                                                decision_times_s=calendar, max_jobs=max_jobs,
                                                max_nodes=max_nodes)
                oracle_by_seed[workload.seed] = {"status": "computed", "result": oracle,
                                                 "decision_times_s": calendar,
                                                 "calendar_definition": "union of selected-policy admission times for this evaluation seed"}
            except OracleSearchLimit as exc:
                oracle_by_seed[workload.seed] = {"status": "search_limit", "upper_bound": exc.upper_bound,
                                                 "nodes": exc.nodes, "decision_times_s": calendar,
                                                 "result": SimpleNamespace(welfare=None, exact=False,
                                                                            upper_bound=exc.upper_bound,
                                                                            algorithm="search_limit_bound",
                                                                            nodes=exc.nodes),
                                                 "calendar_definition": "union of selected-policy admission times for this evaluation seed"}
    for name, rows in eval_rows_by_policy.items():
        if calibration_only:
            continue
        for row, workload in zip(rows, eval_workloads):
            oracle_row = oracle_by_seed.get(workload.seed, {})
            oracle = oracle_row.get("result")
            metrics = row["metrics"]
            if oracle is not None:
                _attach_oracle_metrics(metrics, oracle)
            row["oracle_status"] = oracle_row.get("status", "disabled")
            # Full shared calendars live once under report.oracle; repeating
            # them on every policy row bloats reports without adding data.
            row["oracle_calendar_seed"] = workload.seed
        policies[name]["evaluation"] = rows
        policies[name]["evaluation_request_rows"] = eval_request_rows[name]
        feasible = bool(rows) and all(x["metrics"]["slo"]["all_segments_feasible"] for x in rows)
        policies[name]["evaluation_all_seeds_feasible"] = feasible
        policies[name]["report_status"] = "evaluated_slo_pass" if feasible else ("evaluated_slo_failure" if rows else "no_calibration_feasible_candidate")
    comparable = []
    if not calibration_only:
        for name, policy in policies.items():
            if policy["report_status"] == "evaluated_slo_pass":
                values = [row["metrics"]["revenue_per_gpu_hour"] for row in policy["evaluation"]
                          if row["metrics"]["revenue_per_gpu_hour"] is not None]
                comparable.append({"policy": name,
                                   "mean_revenue_per_gpu_hour": sum(values) / len(values) if values else None,
                                   "parameters": policy["selected_parameters"]})
        comparable.sort(key=lambda item: item["mean_revenue_per_gpu_hour"] or 0, reverse=True)
    report = {"schema_version": 1, "experiment": config.get("name", "experiment"),
            "synthetic": True, "server": asdict(server), "demand": asdict(demand),
            "economics": {"marginal_cost_per_token": (config.get("economics") or {}).get("marginal_cost_per_token"),
                          "status": "synthetic estimate" if config.get("economics") else "unspecified; no reserve cost floor"},
            "calibration_only": calibration_only,
            "horizon_s": horizon, "seeds": seeds, "low_load_reference": references,
            "low_load_reference_config": {"demand": asdict(reference_demand),
                                          "calibration_seeds": seeds["calibration"],
                                          "policy": "affordable FIFO, reserve=1e-12"},
            "oracle": {"enabled": oracle_enabled, "by_evaluation_seed": {
                str(seed): {**{k: v for k, v in row.items() if k != "result"},
                            **({"welfare": row["result"].welfare, "exact": row["result"].exact,
                                "upper_bound": row["result"].upper_bound,
                                "algorithm": row["result"].algorithm, "nodes": row["result"].nodes}
                               if row.get("result") is not None else {})}
                for seed, row in oracle_by_seed.items()}},
                  "slo": {"fill_rate": fill_slo, "ttft_multiplier": 2.0, "e2e_multiplier": 2.0,
                          "required_segments": list(required_segments), "reference_rate_rps": reference_rate_rps},
            "policies": policies,
            "notes": ["Seeded workloads are paired across policies.",
                      "Configurations were selected on calibration seeds only.",
                      "Revenue comparisons are reported only for candidates passing every segment SLO on every calibration seed."]}
    if not calibration_only:
        report["equal_quality_revenue_ranking"] = comparable
        report["ranking_scope"] = "selected configurations passing all required segment SLOs on every calibration and evaluation seed"
    return report
