from dataclasses import replace

from auction_sim.demand import DemandConfig, make_workload


def test_seed_reproduces_full_public_and_private_stream():
    cfg = DemandConfig(rate_rps=2.0, agent_rate_rps=.4)
    a = make_workload(cfg, 17, 30)
    b = make_workload(cfg, 17, 30)
    assert a == b


def test_slack_does_not_change_arrivals_realized_lengths_or_values():
    cfg = DemandConfig(rate_rps=2.0, output_cap_tokens=128)
    a = make_workload(cfg, 6, 20)
    b = make_workload(replace(cfg, reservation_slack=1.75), 6, 20)
    assert a.arrivals == b.arrivals
    assert [(j.actual_output_tokens, j.value_per_token) for j in a.jobs] == [
        (j.actual_output_tokens, j.value_per_token) for j in b.jobs]
    assert all(j.order.max_tokens <= k.order.max_tokens for j, k in zip(a.jobs, b.jobs))


def test_interactive_and_agent_defaults_are_distinct_order_types():
    workload = make_workload(DemandConfig(rate_rps=3, agent_rate_rps=.5), 9, 60)
    kinds = {job.order.segment: job.order.order_type for job in workload.jobs}
    assert kinds["interactive"] == "market"
    assert kinds["agent"] == "limit"
    assert all(job.order.max_wait_s == 0 for job in workload.jobs if job.order.segment == "interactive")
    assert all(job.order.max_wait_s == 300 for job in workload.jobs if job.order.segment == "agent")
    assert all(job.order.bid_per_token == job.value_per_token for job in workload.jobs)
