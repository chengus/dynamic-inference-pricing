"""Run a fixed, paired load study for synthetic pricing mechanisms."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from statistics import mean

import numpy as np
import yaml

from .demand import DemandConfig, make_workload
from .experiments.config import load_config
from .experiments.runner import run_experiment
from .experiments.sweeps import build_sweep_configs


POLICIES = ("uniform_auction", "first_price_hybrid", "flat_price", "static_tiers",
            "time_of_day_grid", "utilization_controller")
SEGMENTS = ("interactive", "agent")


def _avg(values):
    clean = [float(v) for v in values if v is not None]
    return mean(clean) if clean else None


def _paired_bootstrap_interval(differences: list[float], seed: int) -> list[float] | None:
    if not differences:
        return None
    rng = np.random.default_rng(seed)
    samples = np.asarray(differences, dtype=float)
    draws = rng.choice(samples, size=(10_000, len(samples)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def _candidate_diagnostic(policy):
    candidates = policy["calibration_candidates"]
    if not candidates:
        return None
    row = max(candidates, key=lambda c: c["mean_revenue_per_gpu_hour"])
    if row["feasible"]:
        return None
    return {"candidate_id": row["candidate_id"], "parameters": row["parameters"],
            "mean_calibration_revenue_per_gpu_hour": row["mean_revenue_per_gpu_hour"],
            "calibration_seed_count": len(row["seeds"]), "feasible": False,
            "label": "calibration diagnostic only; not selected or evaluated"}


def _cohort_totals(config, seeds):
    demand = DemandConfig(**config.get("demand", {}))
    out = {}
    for seed in seeds:
        workload = make_workload(demand, seed, float(config["horizon_s"]))
        groups = {"all": workload.jobs}
        groups.update({segment: tuple(j for j in workload.jobs if j.order.segment == segment)
                       for segment in SEGMENTS})
        out[str(seed)] = {
            label: {"requests": len(jobs),
                    "prompt_tokens": sum(j.order.prompt_tokens for j in jobs),
                    "actual_output_tokens": sum(j.actual_output_tokens for j in jobs),
                    "total_billable_tokens": sum(j.billable_tokens for j in jobs),
                    "offered_value_upper_bound": sum(j.total_value for j in jobs)}
            for label, jobs in groups.items()}
    return out


def _invoice_replay(summary, tariffs_path: str | Path, output_dir: Path):
    tariffs = json.loads(Path(tariffs_path).read_text(encoding="utf-8"))
    rows = []
    for load, report in summary["runs"].items():
        uniform = report["policies"]["uniform_auction"]
        selected_requests = {seed: request_rows
                             for seed, request_rows in zip(
                                 [row["seed"] for row in uniform["evaluation"]],
                                 uniform.get("evaluation_request_rows", []))}
        offered_cohorts = summary["offered_cohorts_by_load_and_evaluation_seed"][load]
        cohort_counts = {}
        for seed, offered in offered_cohorts.items():
            served = [r for r in selected_requests.get(int(seed), []) if r["status"] == "completed"]
            cohort_counts[(seed, "offered_all_requests")] = {
                "requests": offered["all"]["requests"],
                "prompt_tokens": offered["all"]["prompt_tokens"],
                "actual_output_tokens": offered["all"]["actual_output_tokens"],
                "basis": "all offered evaluation requests, identical numerical token counts across tariffs"}
            if int(seed) in selected_requests:
                cohort_counts[(seed, "uniform_auction_completed")] = {
                    "requests": len(served),
                    "prompt_tokens": sum(r["prompt_tokens"] for r in served),
                    "actual_output_tokens": sum(r["actual_output_tokens"] for r in served),
                    "basis": "requests completed by selected Uniform Auction policy; excludes timeout/infeasible"}
        for (seed, cohort_name), counts in cohort_counts.items():
            for rate in tariffs["rates"]:
                bill = (counts["prompt_tokens"] * rate["input_per_million"] +
                        counts["actual_output_tokens"] * rate["output_per_million"]) / 1_000_000
                rows.append({"load": load, "evaluation_seed": int(seed), "cohort": cohort_name,
                             "provider": rate["provider"], "model": rate["model"],
                             "rate_mode": rate["mode"],
                             "request_count": counts["requests"],
                             "prompt_tokens_uncached": counts["prompt_tokens"],
                             "actual_output_tokens": counts["actual_output_tokens"],
                             "billed_usd_uncached": bill,
                             "input_rate_per_million": rate["input_per_million"],
                             "output_rate_per_million": rate["output_per_million"],
                             "cached_input_rate_per_million_context_only": rate["cached_input_per_million"],
                             "billing_basis": counts["basis"],
                             "tariff_notes": rate["notes"], "source_url": rate["source_url"]})
    aggregate_map = {}
    for row in rows:
        key = (row["load"], row["cohort"], row["provider"], row["model"], row["rate_mode"])
        acc = aggregate_map.setdefault(key, {"bills": [], "input_tokens": [], "output_tokens": []})
        acc["bills"].append(row["billed_usd_uncached"])
        acc["input_tokens"].append(row["prompt_tokens_uncached"])
        acc["output_tokens"].append(row["actual_output_tokens"])
    aggregate_rows = [{"load": key[0], "cohort": key[1], "provider": key[2], "model": key[3],
                       "rate_mode": key[4], "evaluation_seed_count": len(values["bills"]),
                       "mean_bill_usd_per_seed": _avg(values["bills"]),
                       "total_bill_usd_across_seeds": sum(values["bills"]),
                       "mean_prompt_tokens_per_seed": _avg(values["input_tokens"]),
                       "mean_output_tokens_per_seed": _avg(values["output_tokens"])}
                      for key, values in aggregate_map.items()]
    result = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "tariff_file": str(tariffs_path), "tariffs_retrieved_date": tariffs["retrieved_date"],
              "currency": tariffs["currency"], "rate_unit": tariffs["rate_unit"],
              "method": ("Uncached invoice replay using offered prompt and actual output token totals. "
                         "No cache hits are assumed; cached input rates are context only. "
                         "Token counts are identical numerically across providers; provider-specific "
                         "tokenizers and actual text are not replayed."),
              "comparability_note": ("Invoices are pricing context for a fixed synthetic request cohort, not a quality, "
                                     "latency, service-level, or revenue comparison. Offered-cohort bills must not be "
                                     "directly divided by auction revenue from a different served cohort."),
              "tariff_assumptions": tariffs["replay_assumptions"],
              "aggregates": aggregate_rows, "rows": rows}
    (output_dir / "provider-invoice-replay.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "provider-invoice-replay.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "provider-invoice-summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0]) if aggregate_rows else [])
        writer.writeheader()
        writer.writerows(aggregate_rows)
    return result


def _summary_row(load, report, policy_name, cohort):
    policy = report["policies"][policy_name]
    evaluations = policy["evaluation"]
    common_elapsed = {}
    for row in evaluations:
        seed = row["seed"]
        common_elapsed[seed] = max(
            other["metrics"]["elapsed_s_including_drain"]
            for selected in report["policies"].values()
            for other in selected["evaluation"] if other["seed"] == seed)
    rows = {"load": load, "policy": policy_name,
            "calibration_feasible_candidate_count": policy["calibration_feasible_candidates"],
            "selected_parameters": json.dumps(policy["selected_parameters"], sort_keys=True),
            "evaluation_count": len(evaluations), "status": policy["report_status"],
            "calibration_diagnostic": json.dumps(_candidate_diagnostic(policy), sort_keys=True)}
    metric_fields = ("revenue_per_gpu_hour", "fill_rate", "welfare", "buyer_surplus",
                     "busy_utilization", "time_weighted_kv_utilization", "timeout_rate")
    for field in metric_fields:
        rows[f"mean_{field}"] = _avg([r["metrics"].get(field) for r in evaluations])
    rows["mean_common_exposure_revenue_per_gpu_hour"] = _avg([
        r["metrics"]["revenue"] * 3600 / common_elapsed[r["seed"]] for r in evaluations])
    rows["mean_fixed_arrival_horizon_revenue_per_gpu_hour"] = _avg([
        r["metrics"]["revenue"] * 3600 / r["metrics"]["arrival_horizon_s"] for r in evaluations])
    rows["mean_served_value_offered_bound_ratio_lower_bound"] = _avg([
        r["metrics"]["welfare"] /
        cohort[str(r["seed"])]["all"]["offered_value_upper_bound"]
        if cohort[str(r["seed"])]["all"]["offered_value_upper_bound"] else None
        for r in evaluations])
    for segment in SEGMENTS:
        key = f"segment:{segment}"
        for metric in ("fill_rate", "ttft_p95_s", "e2e_p95_s", "welfare", "buyer_surplus", "timeout_rate"):
            rows[f"mean_{segment}_{metric}"] = _avg([
                r["metrics"]["by_group"][key].get(metric) for r in evaluations])
    return rows


def run_study(config_path: str | Path, output_dir: str | Path,
              tariffs_path: str | Path = "reports/provider-pricing-2026-10-27.json") -> dict:
    base = load_config(config_path)
    if float(base.get("horizon_s", 0)) != 600:
        raise ValueError("provider study requires a 600 second horizon")
    if base["seeds"] != {"calibration": [0, 1, 2],
                         "evaluation": [100, 101, 102, 103, 104]}:
        raise ValueError("provider study requires calibration seeds [0,1,2] and evaluation seeds [100..104]")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    command = ("UV_CACHE_DIR=/private/tmp/auction-pricing-uv-cache uv run --locked --offline "
               f"python -m auction_sim.provider_study --config {config_path} --output-dir {output_dir} "
               "--allow-simulation")
    reports, summary_rows, cohorts = {}, [], {}
    derived_configs = build_sweep_configs(base, "load", ["low", "medium", "high"])
    for config in derived_configs:
        label = config["sweep"]["label"]
        with (output / f"{label}-config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        cohort = _cohort_totals(config, config["seeds"]["evaluation"])
        report = run_experiment(config)
        report["sweep"] = config["sweep"]
        reports[label] = report
        cohorts[label] = cohort
        for name in POLICIES:
            summary_rows.append(_summary_row(label, report, name, cohort))

    flat_differences = {}
    for load, report in reports.items():
        flat = report["policies"]["flat_price"]
        if flat["report_status"] != "evaluated_slo_pass":
            continue
        common_elapsed = {r["seed"]: max(
            other["metrics"]["elapsed_s_including_drain"]
            for selected in report["policies"].values()
            for other in selected["evaluation"] if other["seed"] == r["seed"])
            for r in flat["evaluation"]}
        flat_by_seed = {r["seed"]: r["metrics"]["revenue"] * 3600 / common_elapsed[r["seed"]]
                        for r in flat["evaluation"]}
        for policy_name in ("uniform_auction", "first_price_hybrid"):
            policy = report["policies"][policy_name]
            diffs = []
            if policy["report_status"] == "evaluated_slo_pass":
                for row in policy["evaluation"]:
                    if row["seed"] in flat_by_seed:
                        diffs.append(row["metrics"]["revenue"] * 3600 / common_elapsed[row["seed"]]
                                     - flat_by_seed[row["seed"]])
            interval = _paired_bootstrap_interval(diffs, 261027 + sum(map(ord, load + policy_name)))
            auction_mean = _avg([r["metrics"]["revenue"] * 3600 / common_elapsed[r["seed"]]
                                 for r in policy["evaluation"]]) if policy["report_status"] == "evaluated_slo_pass" else None
            flat_mean = _avg(flat_by_seed.values())
            flat_differences[f"{load}/{policy_name}"] = {
                "status": "valid paired equal-SLO comparison" if interval is not None else
                          "no qualified equal-quality comparison",
                "paired_seed_count": len(diffs),
                "mean_auction_minus_flat_revenue_per_gpu_hour": _avg(diffs),
                "mean_common_exposure_revenue_per_gpu_hour": auction_mean,
                "flat_common_exposure_revenue_per_gpu_hour": flat_mean,
                "auction_to_flat_common_exposure_rate_ratio": auction_mean / flat_mean
                if auction_mean is not None and flat_mean else None,
                "bootstrap_95_percent_interval": interval,
                "bootstrap_resamples": 10_000,
                "baseline": "selected flat_price candidate; must pass all calibration and evaluation SLOs",
                "exposure_denominator": "for each load/seed, the maximum elapsed_s_including_drain across all evaluated policies",
            }

    summary = {
        "schema_version": 1,
        "experiment": "controlled synthetic pricing provider comparison",
        "created_at_utc": timestamp,
        "source_config": str(config_path),
        "command": command,
        "study_design": {
            "horizon_s": 600, "load_labels": ["low", "medium", "high"],
            "only_swept_parameter": "arrival rates; the existing load builder scales agent MMPP rate proportionally",
            "calibration_seeds": base["seeds"]["calibration"],
            "evaluation_seeds": base["seeds"]["evaluation"],
            "policies": list(POLICIES), "economics": base.get("economics"),
            "slo": "95% fill, with unchanged 2x low-load reference TTFT and E2E limits, per required segment",
            "warning": "Synthetic workloads, values, service curve, and reserve economics; revenue is not comparable to real provider prices.",
        },
        "paired_auction_vs_best_feasible_flat": flat_differences,
        "offered_cohorts_by_load_and_evaluation_seed": cohorts,
        "runs": reports,
    }
    _invoice_replay(summary, tariffs_path, output)
    with gzip.open(output / "raw-seed-results.json.gz", "wt", encoding="utf-8") as f:
        json.dump(summary, f, sort_keys=True, separators=(",", ":"))
    summary["policy_summary_rows"] = summary_rows
    summary.pop("runs")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = list(summary_rows[0]) if summary_rows else []
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary


def reaggregate_saved(output_dir: str | Path,
                     tariffs_path: str | Path = "reports/provider-pricing-2026-10-27.json") -> dict:
    """Recompute summaries/invoices from saved seed results without simulation."""
    output = Path(output_dir)
    path = output / "summary.json"
    compact = json.loads(path.read_text(encoding="utf-8"))
    raw_path = output / "raw-seed-results.json.gz"
    if raw_path.exists():
        with gzip.open(raw_path, "rt", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = compact
    rows = []
    for load, report in summary["runs"].items():
        cohort = summary["offered_cohorts_by_load_and_evaluation_seed"][load]
        for name in POLICIES:
            rows.append(_summary_row(load, report, name, cohort))
        for policy in report["policies"].values():
            for row in policy["evaluation"]:
                row.pop("oracle_decision_times_s", None)
                if row.get("oracle_status") not in (None, "disabled"):
                    row["oracle_calendar_seed"] = row["seed"]

    flat_differences = {}
    for load, report in summary["runs"].items():
        flat = report["policies"]["flat_price"]
        if flat["report_status"] != "evaluated_slo_pass":
            continue
        common_elapsed = {r["seed"]: max(
            other["metrics"]["elapsed_s_including_drain"]
            for selected in report["policies"].values()
            for other in selected["evaluation"] if other["seed"] == r["seed"])
            for r in flat["evaluation"]}
        flat_by_seed = {r["seed"]: r["metrics"]["revenue"] * 3600 / common_elapsed[r["seed"]]
                        for r in flat["evaluation"]}
        for policy_name in ("uniform_auction", "first_price_hybrid"):
            policy = report["policies"][policy_name]
            diffs = ([r["metrics"]["revenue"] * 3600 / common_elapsed[r["seed"]] - flat_by_seed[r["seed"]]
                      for r in policy["evaluation"] if r["seed"] in flat_by_seed]
                     if policy["report_status"] == "evaluated_slo_pass" else [])
            auction_mean = _avg([r["metrics"]["revenue"] * 3600 / common_elapsed[r["seed"]]
                                 for r in policy["evaluation"]]) if policy["report_status"] == "evaluated_slo_pass" else None
            flat_mean = _avg(flat_by_seed.values())
            interval = _paired_bootstrap_interval(diffs, 261027 + sum(map(ord, load + policy_name)))
            flat_differences[f"{load}/{policy_name}"] = {
                "status": "valid paired equal-SLO comparison" if interval is not None else
                          "no qualified equal-quality comparison",
                "paired_seed_count": len(diffs),
                "mean_auction_minus_flat_revenue_per_gpu_hour": _avg(diffs),
                "mean_common_exposure_revenue_per_gpu_hour": auction_mean,
                "flat_common_exposure_revenue_per_gpu_hour": flat_mean,
                "auction_to_flat_common_exposure_rate_ratio": auction_mean / flat_mean
                if auction_mean is not None and flat_mean else None,
                "bootstrap_95_percent_interval": interval,
                "bootstrap_resamples": 10_000,
                "baseline": "selected flat_price candidate; must pass all calibration and evaluation SLOs",
                "exposure_denominator": "for each load/seed, the maximum elapsed_s_including_drain across all evaluated policies",
            }
    summary.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
    summary["analysis_updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["paired_auction_vs_best_feasible_flat"] = flat_differences
    summary["study_design"]["exposure_rate_methods"] = {
        "common_exposure": "Each evaluated policy's revenue divided by the maximum per-seed elapsed_s_including_drain across all evaluated policies.",
        "fixed_arrival_horizon": "Each evaluated policy's revenue divided by the 600 second arrival horizon.",
        "original_revenue_per_gpu_hour": "Preserved from simulator metrics; divides by policy-specific elapsed time including drain and is reported as a diagnostic because policies can extend drain differently.",
    }
    summary["policy_summary_rows"] = rows
    _invoice_replay(summary, tariffs_path, output)
    with gzip.open(raw_path, "wt", encoding="utf-8") as f:
        json.dump(summary, f, sort_keys=True, separators=(",", ":"))
    summary.pop("runs")
    summary.pop("offered_cohorts_by_load_and_evaluation_seed")
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/provider-study.yaml")
    parser.add_argument("--output-dir", default="results/provider-comparison-2026-10-27")
    parser.add_argument("--tariffs", default="reports/provider-pricing-2026-10-27.json")
    parser.add_argument("--reaggregate-existing", action="store_true",
                        help="refresh summary and invoice files from saved run results without simulation")
    parser.add_argument("--allow-simulation", action="store_true",
                        help="acknowledge that the local synthetic simulator will run")
    args = parser.parse_args(argv)
    if args.reaggregate_existing:
        summary = reaggregate_saved(args.output_dir, args.tariffs)
        print(f"Refreshed saved results for {len(summary['policy_summary_rows']) // len(POLICIES)} load levels; no simulation run")
        return 0
    if not args.allow_simulation:
        parser.error("provider study runs simulations; pass --allow-simulation explicitly")
    summary = run_study(args.config, args.output_dir, args.tariffs)
    print(f"Saved study results for {len(summary['study_design']['load_labels'])} load levels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
