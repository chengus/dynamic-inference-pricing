"""Exercise harness wiring using fixtures, without running the event engine."""

from pathlib import Path
from dataclasses import replace
import gc
from types import SimpleNamespace
import weakref

from auction_sim.demand import Workload
from auction_sim.experiments.config import load_config
from auction_sim.experiments import runner
from auction_sim.models import Job, Order, PricePoint, RequestResult, SimulationResult, UsageInterval


def fixture_harness(monkeypatch):
    seeds_seen, policies_seen = [], []

    def workload(config, seed, horizon):
        seeds_seen.append(seed)
        jobs = tuple(Job(Order(f"{seed}-{segment}", arrival, 1, 1, 1.0, wait,
                               segment, kind), 1, 1.0)
                     for segment, arrival, wait, kind in [
                         ("interactive", 0.0, 0.0, "market"),
                         ("agent", 0.5, 5.0, "limit")])
        return Workload(jobs, seed, horizon, tuple(j.order.arrival_s for j in jobs), {})

    class FixtureSimulator:
        def __init__(self, server):
            pass

        def run(self, jobs, policy, horizon):
            policies_seen.append(policy)
            # This is a fixed result fixture, with no queue or service execution.
            requests = tuple(RequestResult(j, "completed", j.order.arrival_s,
                                           j.order.arrival_s + .1,
                                           j.order.arrival_s + .2, .1) for j in jobs)
            return SimulationResult(requests, (), (), horizon, horizon)

    monkeypatch.setattr(runner, "make_workload", workload)
    monkeypatch.setattr(runner, "Simulator", FixtureSimulator)
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    config["seeds"] = {"calibration": [0], "evaluation": [100]}
    config["horizon_s"] = 2
    config["oracle"] = {"enabled": False}
    return config, seeds_seen, policies_seen


def test_calibration_only_never_generates_evaluation_workloads(monkeypatch):
    config, seeds_seen, policies_seen = fixture_harness(monkeypatch)
    report = runner.run_experiment(config, calibration_only=True)
    assert set(seeds_seen) == {0}
    assert report["calibration_only"]
    assert not report["oracle"]["enabled"]
    assert all(row["report_status"] == "calibrated_only" for row in report["policies"].values())
    assert all(not row["evaluation"] for row in report["policies"].values())
    assert len({id(policy) for policy in policies_seen}) == len(policies_seen)


def test_evaluation_streams_full_results_but_preserves_compact_metrics(monkeypatch):
    config, _, _ = fixture_harness(monkeypatch)
    refs = []
    original = runner.Simulator.run

    def traced_run(self, jobs, policy, horizon):
        # At most the immediately previous result can be live while Python
        # evaluates the next run() call. Accumulating eval_raw would exceed 1.
        assert sum(ref() is not None for ref in refs) <= 1
        result = original(self, jobs, policy, horizon)
        result = replace(result,
                         usage=(UsageInterval(0.0, .1, True, 1, 1, 2),),
                         prices=(PricePoint(0.0, .25, 1, "clearing"),))
        refs.append(weakref.ref(result))
        return result

    monkeypatch.setattr(runner.Simulator, "run", traced_run)
    report = runner.run_experiment(config)
    gc.collect()
    assert all(ref() is None for ref in refs)
    policy = report["policies"]["uniform_auction"]
    metrics = policy["evaluation"][0]["metrics"]
    assert metrics["price_trade_count"] == 1
    assert metrics["busy_utilization"] > 0
    assert metrics["by_group"]["all"]["completed"] == 2
    assert len(policy["evaluation_request_rows"][0]) == 2
    assert policy["evaluation_request_rows"][0][0]["actual_output_tokens"] == 1


def test_frozen_evaluation_uses_one_shared_oracle_per_seed(monkeypatch):
    config, seeds_seen, _ = fixture_harness(monkeypatch)
    config["oracle"] = {"enabled": True, "max_jobs": 6, "max_nodes": 100}
    calls = []

    def fixture_oracle(jobs, server, horizon, **kwargs):
        calls.append(kwargs)
        welfare = sum(job.total_value for job in jobs)
        return SimpleNamespace(welfare=welfare, exact=True, upper_bound=welfare,
                               algorithm="fixed test fixture; no search", nodes=1)

    monkeypatch.setattr("auction_sim.oracle.exact_hindsight_oracle", fixture_oracle)
    report = runner.run_experiment(config)
    assert set(seeds_seen) == {0, 100}
    assert len(calls) == 1
    assert calls[0]["decision_times_s"] == [0.0, 0.5]
    assert len(report["equal_quality_revenue_ranking"]) == 6
    for policy in report["policies"].values():
        assert policy["evaluation_all_seeds_feasible"]
        assert policy["evaluation"][0]["metrics"]["oracle"]["efficiency_ratio"] == 1
        assert policy["evaluation"][0]["oracle_calendar_seed"] == 100
        assert "oracle_decision_times_s" not in policy["evaluation"][0]
    assert report["oracle"]["by_evaluation_seed"]["100"]["decision_times_s"] == [0.0, 0.5]


def test_evaluation_slo_failure_is_excluded_from_revenue_ranking(monkeypatch):
    config, _, _ = fixture_harness(monkeypatch)
    original = runner.Simulator.run

    def fixture_with_failure(self, jobs, policy, horizon):
        result = original(self, jobs, policy, horizon)
        if jobs[0].order.request_id.startswith("100-") and type(policy).__name__ == "UniformAuction":
            rows = tuple(replace(r, status="timeout", admitted_at_s=None,
                                 first_token_at_s=None, completed_at_s=None,
                                 locked_rate_per_token=None)
                         if r.job.order.segment == "agent" else r for r in result.requests)
            return replace(result, requests=rows)
        return result

    monkeypatch.setattr(runner.Simulator, "run", fixture_with_failure)
    report = runner.run_experiment(config)
    assert report["policies"]["uniform_auction"]["report_status"] == "evaluated_slo_failure"
    assert "uniform_auction" not in {r["policy"] for r in report["equal_quality_revenue_ranking"]}
