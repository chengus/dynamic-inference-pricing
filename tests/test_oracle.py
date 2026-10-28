from auction_sim.models import Job, Order, ServerConfig
from auction_sim.oracle import OracleSearchLimit, exact_hindsight_oracle


def make_job(rid, value, *, arrival=0, wait=0):
    return Job(Order(rid, arrival, 1, 1, 10, wait, "test"), 1, value)


def test_oracle_selects_high_value_job_when_zero_wait_orders_compete():
    jobs = [make_job("low", 2), make_job("high", 10)]
    result = exact_hindsight_oracle(
        jobs, ServerConfig(t0_s=1, k_s=0, prefill_s_per_token=0, max_batch_size=1), 0
    )
    assert result.exact
    assert result.welfare == 20
    assert result.welfare <= result.upper_bound
    assert result.algorithm.startswith("exhaustive")


def test_oracle_can_wait_for_deadline_tie_after_first_request_releases_slot():
    jobs = [make_job("first", 5, wait=1), make_job("second", 10, wait=1)]
    result = exact_hindsight_oracle(
        jobs, ServerConfig(t0_s=1, k_s=0, prefill_s_per_token=0, max_batch_size=1), 0
    )
    assert result.exact
    assert result.welfare == 30


def test_oracle_refuses_to_claim_exactness_after_size_limit():
    jobs = [make_job(str(i), 1) for i in range(3)]
    try:
        exact_hindsight_oracle(jobs, ServerConfig(), 0, max_jobs=2)
    except OracleSearchLimit as exc:
        assert exc.upper_bound == sum(job.total_value for job in jobs)
    else:
        raise AssertionError("expected OracleSearchLimit")


def test_oracle_search_budget_raises_with_certified_offered_value_bound():
    jobs = [make_job(str(i), 1) for i in range(3)]
    try:
        exact_hindsight_oracle(jobs, ServerConfig(), 0, max_nodes=1)
    except OracleSearchLimit as exc:
        assert exc.upper_bound == sum(job.total_value for job in jobs)
    else:
        raise AssertionError("expected OracleSearchLimit")


def test_oracle_enumerates_prefill_order_not_just_admitted_sets():
    # Reversing a,b frees b's slot at t=2, admitting c before its zero-wait expiry.
    jobs = [Job(Order("a", 0, 2, 1, 1, 0, "test"), 1, 1),
            Job(Order("b", 0, 1, 1, 1, 0, "test"), 1, 1),
            Job(Order("c", 2, 1, 1, 1, 0, "test"), 1, 1)]
    server = ServerConfig(t0_s=1, k_s=0, prefill_s_per_token=1,
                          prefill_chunk_tokens=1, max_batch_size=2, max_kv_tokens=8)
    result = exact_hindsight_oracle(jobs, server, 2)
    assert result.exact
    assert result.welfare == sum(job.total_value for job in jobs)
