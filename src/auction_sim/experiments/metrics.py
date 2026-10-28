"""Accounting and service diagnostics for a completed simulation result."""

from __future__ import annotations

import math
from typing import Any, Mapping

from ..models import ServerConfig, SimulationResult


def _quantile(values: list[float], q: float) -> float | None:
    return float(sorted(values)[max(0, math.ceil(q * len(values)) - 1)]) if values else None


def summarize(result: SimulationResult, server: ServerConfig, *,
              low_load_reference: Mapping[str, Mapping[str, float]] | None = None,
              oracle: Any | None = None, fill_slo: float = 0.95,
              ttft_multiplier: float = 2.0, e2e_multiplier: float = 2.0,
              required_segments: tuple[str, ...] = ()) -> dict[str, Any]:
    """Return JSON-friendly metrics; the drain time is included in revenue/GPU-hour."""
    requests = result.requests
    completed = [r for r in requests if r.status == "completed"]
    segments = sorted({r.job.order.segment for r in requests} | set(required_segments))
    order_types = sorted({r.job.order.order_type for r in requests})
    groups = {"all": requests}
    groups.update({f"segment:{s}": [r for r in requests if r.job.order.segment == s] for s in segments})
    groups.update({f"order_type:{t}": [r for r in requests if r.job.order.order_type == t] for t in order_types})

    def group_metrics(group):
        done = [r for r in group if r.status == "completed"]
        ttft = [r.first_token_at_s - r.job.order.arrival_s for r in done if r.first_token_at_s is not None]
        e2e = [r.completed_at_s - r.job.order.arrival_s for r in done if r.completed_at_s is not None]
        tpot = [(r.completed_at_s - r.first_token_at_s) / (r.job.actual_output_tokens - 1)
                for r in done if r.first_token_at_s is not None and r.completed_at_s is not None
                and r.job.actual_output_tokens > 1]
        total = len(group)
        timeout = sum(r.status == "timeout" for r in group)
        infeasible = sum(r.status == "infeasible" for r in group)
        waits = [r.admitted_at_s - r.job.order.arrival_s for r in done if r.admitted_at_s is not None]
        return {"requests": total, "completed": len(done), "fill_rate": len(done) / total if total else None,
                "timeout": timeout, "timeout_rate": timeout / total if total else None,
                "infeasible": infeasible, "infeasible_rate": infeasible / total if total else None,
                "revenue": sum(r.revenue for r in done),
                "buyer_surplus": sum(r.job.total_value - r.revenue for r in done),
                "welfare": sum(r.job.total_value for r in done),
                "admission_wait_p50_s": _quantile(waits, .50), "admission_wait_p95_s": _quantile(waits, .95),
                "ttft_p50_s": _quantile(ttft, .50), "ttft_p95_s": _quantile(ttft, .95),
                "e2e_p50_s": _quantile(e2e, .50), "e2e_p95_s": _quantile(e2e, .95),
                "tpot_p50_s": _quantile(tpot, .50), "tpot_p95_s": _quantile(tpot, .95)}

    gm = {name: group_metrics(group) for name, group in groups.items()}
    elapsed = result.elapsed_s
    total_revenue = sum(r.revenue for r in completed)
    gpu_hours = elapsed / 3600.0
    busy_s = sum(x.end_s - x.start_s for x in result.usage if x.busy)
    kv_time = sum((x.end_s - x.start_s) * x.reserved_kv_tokens for x in result.usage)
    clearing_points = [p for p in result.prices if p.kind == "clearing"]
    price_values = [p.price_per_token for p in clearing_points]
    price_mean = sum(price_values) / len(price_values) if price_values else None
    price_std = (math.sqrt(sum((p - price_mean) ** 2 for p in price_values) / len(price_values))
                 if price_values and price_mean is not None else None)
    # Reservation waste is capacity held for output that was never generated.
    reserved_output = sum(r.job.order.max_tokens for r in requests if r.status == "completed")
    used_output = sum(r.job.actual_output_tokens for r in completed)
    waste = reserved_output - used_output
    metrics: dict[str, Any] = {
        "requests": len(requests), "completed": len(completed),
        "timeout": sum(r.status == "timeout" for r in requests),
        "infeasible": sum(r.status == "infeasible" for r in requests),
        "arrival_horizon_s": result.arrival_horizon_s, "elapsed_s_including_drain": elapsed,
        "revenue": total_revenue, "revenue_per_gpu_hour": total_revenue / gpu_hours if gpu_hours else None,
        "welfare": gm["all"]["welfare"], "buyer_surplus": gm["all"]["buyer_surplus"],
        "fill_rate": gm["all"]["fill_rate"],
        "output_tokens_per_second": used_output / elapsed if elapsed else None,
        "busy_utilization": busy_s / elapsed if elapsed else None,
        "time_weighted_kv_utilization": kv_time / (elapsed * server.max_kv_tokens) if elapsed else None,
        "reservation_waste_output_tokens": waste,
        "reservation_waste_fraction": waste / reserved_output if reserved_output else None,
        "price_trade_count": len(price_values), "clearing_price_mean": price_mean,
        "clearing_price_std": price_std, "clearing_price_cv": price_std / price_mean if price_std is not None and price_mean else None,
        "clearing_price_series": [{"time_s": p.time_s, "price_per_token": p.price_per_token,
                                   "admissions": p.admissions} for p in clearing_points],
        "admission_price_series": [{"time_s": p.time_s, "price_per_token": p.price_per_token,
                                     "admissions": p.admissions, "kind": p.kind} for p in result.prices],
        "by_group": gm,
        "slo": {},
    }
    for segment in segments:
        m = gm[f"segment:{segment}"]
        ref = (low_load_reference or {}).get(segment)
        thresholds = ({"ttft_p95_s": ref.get("ttft_p95_s") * ttft_multiplier,
                       "e2e_p95_s": ref.get("e2e_p95_s") * e2e_multiplier}
                      if ref and ref.get("ttft_p95_s") is not None and ref.get("e2e_p95_s") is not None else None)
        has_completions = m["completed"] > 0
        done = [r for r in requests if r.job.order.segment == segment and r.status == "completed"]
        ttft_values = [r.first_token_at_s - r.job.order.arrival_s for r in done if r.first_token_at_s is not None]
        e2e_values = [r.completed_at_s - r.job.order.arrival_s for r in done if r.completed_at_s is not None]
        ttft_limit = thresholds["ttft_p95_s"] if thresholds else None
        e2e_limit = thresholds["e2e_p95_s"] if thresholds else None
        m["ttft_slo_exceedance_fraction"] = (sum(x > ttft_limit for x in ttft_values) / len(ttft_values)
                                                if ttft_values and ttft_limit is not None else None)
        m["e2e_slo_exceedance_fraction"] = (sum(x > e2e_limit for x in e2e_values) / len(e2e_values)
                                               if e2e_values and e2e_limit is not None else None)
        checks = {"fill": m["fill_rate"] is not None and m["fill_rate"] >= fill_slo,
                  "ttft": bool(has_completions and thresholds and m["ttft_p95_s"] is not None and m["ttft_p95_s"] <= thresholds["ttft_p95_s"]),
                  "e2e": bool(has_completions and thresholds and m["e2e_p95_s"] is not None and m["e2e_p95_s"] <= thresholds["e2e_p95_s"])}
        metrics["slo"][segment] = {"thresholds": thresholds, "checks": checks,
                                    "feasible": bool(all(checks.values())) if thresholds else False,
                                    "reference_available": thresholds is not None,
                                    "zero_completions": not has_completions}
    elapsed = result.elapsed_s
    total = len(requests)
    metrics["timeout_rate"] = metrics["timeout"] / total if total else None
    metrics["infeasible_rate"] = metrics["infeasible"] / total if total else None
    admission_waits = [r.admitted_at_s - r.job.order.arrival_s for r in requests if r.admitted_at_s is not None]
    metrics["admission_wait_p50_s"] = _quantile(admission_waits, .50)
    metrics["admission_wait_p95_s"] = _quantile(admission_waits, .95)
    active_slot_time = sum((x.end_s - x.start_s) * x.active_requests for x in result.usage)
    decoding_slot_time = sum((x.end_s - x.start_s) * x.decoding_requests for x in result.usage)
    metrics["time_weighted_batch_utilization"] = active_slot_time / (elapsed * server.max_batch_size) if elapsed else None
    metrics["time_weighted_decoding_slot_utilization"] = decoding_slot_time / (elapsed * server.max_batch_size) if elapsed else None
    metrics["peak_active_requests"] = max((x.active_requests for x in result.usage), default=0)
    metrics["peak_decoding_requests"] = max((x.decoding_requests for x in result.usage), default=0)
    metrics["slo"]["all_segments_feasible"] = bool(segments) and all(
        metrics["slo"][s]["feasible"] for s in segments)
    if oracle is not None:
        oracle_welfare = getattr(oracle, "welfare", None)
        exact = bool(getattr(oracle, "exact", False))
        bound = getattr(oracle, "upper_bound", None)
        welfare = gm["all"]["welfare"]
        exact_ratio = welfare / oracle_welfare if exact and oracle_welfare else None
        lower_ratio = welfare / bound if not exact and bound else None
        metrics["oracle"] = {"exact": exact, "welfare": oracle_welfare,
                              "upper_bound": bound,
                              "algorithm": getattr(oracle, "algorithm", None),
                              "nodes": getattr(oracle, "nodes", None),
                              "efficiency_ratio": exact_ratio,
                              "certified_lower_ratio": lower_ratio,
                              "efficiency_interval": ([exact_ratio, exact_ratio] if exact_ratio is not None else
                                                       [lower_ratio, 1.0] if lower_ratio is not None else None),
                              "interpretation": "exact optimum ratio on supplied decision calendar" if exact else
                              ("certified lower ratio to supplied upper bound" if bound else "no certified ratio")}
    return metrics
