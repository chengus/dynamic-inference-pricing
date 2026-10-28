from auction_sim.experiments.metrics import summarize
from auction_sim.models import Job, Order, PricePoint, RequestResult, ServerConfig, SimulationResult, UsageInterval


def test_full_drain_denominator_and_zero_completion_slo():
    order = Order("r1", 0, 4, 8, .2, 0, "standard", "market")
    job = Job(order, 2, .2)
    result = SimulationResult((RequestResult(job, "timeout"),),
                              (UsageInterval(0, 2, True, 1, 1, 12),), (), 1, 2)
    m = summarize(result, ServerConfig(), low_load_reference={"standard": {
        "ttft_p95_s": 1.0, "e2e_p95_s": 2.0}})
    assert m["elapsed_s_including_drain"] == 2
    assert m["revenue_per_gpu_hour"] == 0
    assert m["slo"]["standard"]["zero_completions"] is True
    assert m["slo"]["standard"]["feasible"] is False


def test_oracle_labels_exact_and_bounded_ratios_separately():
    job = Job(Order("r1", 0, 4, 8, .2, 0, "standard", "market"), 2, .2)
    result = SimulationResult((RequestResult(job, "completed", 0, 1, 3, .1),), (), (), 1, 3)
    class Oracle:
        welfare, exact, upper_bound = 5.0, False, 10.0
    m = summarize(result, ServerConfig(), oracle=Oracle())
    assert m["oracle"]["efficiency_ratio"] is None
    assert m["oracle"]["certified_lower_ratio"] == m["by_group"]["all"]["welfare"] / 10


def test_surplus_is_value_minus_charge_without_clamping():
    job = Job(Order("a", 0, 1, 1, 10, 0, "interactive"), 1, 1)
    result = SimulationResult((RequestResult(job, "completed", 0, 1, 1, 5),), (), (), 1, 1)
    metrics = summarize(result, ServerConfig())
    assert metrics["buyer_surplus"] == -8
    assert metrics["welfare"] - metrics["revenue"] == metrics["buyer_surplus"]


def test_tier_mean_admission_prices_are_not_called_uniform_clearing_prices():
    result = SimulationResult((), (), (PricePoint(0, 2, 2, "mean_admission"),), 1, 1)
    metrics = summarize(result, ServerConfig())
    assert metrics["clearing_price_series"] == []
    assert metrics["clearing_price_mean"] is None
    assert metrics["admission_price_series"][0]["price_per_token"] == 2


def test_missing_required_segment_cannot_pass_service_constraints():
    job = Job(Order("a", 0, 1, 1, 1, 0, "interactive"), 1, 1)
    result = SimulationResult((RequestResult(job, "completed", 0, .1, .1, 1),), (), (), 1, 1)
    metrics = summarize(result, ServerConfig(), required_segments=("interactive", "agent"),
                        low_load_reference={"interactive": {"ttft_p95_s": 1, "e2e_p95_s": 1},
                                            "agent": {"ttft_p95_s": 1, "e2e_p95_s": 1}})
    assert not metrics["slo"]["all_segments_feasible"]
    assert not metrics["slo"]["agent"]["checks"]["fill"]
