import pytest

from auction_sim.engine import PolicyError, Simulator
from auction_sim.models import Admission, Decision, Job, Order, ServerConfig


def job(rid="a", *, arrival=0.0, prompt=1, maximum=2, actual=1, bid=5.0, value=10.0, wait=0.0):
    return Job(Order(rid, arrival, prompt, maximum, bid, wait, "test"), actual, value)


class AdmitAll:
    def __init__(self, rate=None):
        self.rate = rate
        self.seen = []
        self.intervals = []

    def step(self, pending, free_capacity, t):
        self.seen.append((tuple(pending), free_capacity, t))
        choices = []
        for order in pending:
            if free_capacity.batch_slots <= len(choices):
                break
            if order.reserved_kv_tokens <= free_capacity.kv_tokens - sum(
                next(x for x in pending if x.request_id == choice.request_id).reserved_kv_tokens
                for choice in choices
            ):
                choices.append(Admission(order.request_id, order.bid_per_token if self.rate is None else self.rate))
        return Decision(tuple(choices))

    def observe(self, interval):
        self.intervals.append(interval)

    def next_wakeup(self, t):
        return None


def test_actual_tokens_set_completion_and_locked_price_bills_actual_use():
    policy = AdmitAll(rate=1.25)
    result = Simulator(ServerConfig(t0_s=1, k_s=0.5, prefill_s_per_token=0.2)).run(
        [job(prompt=2, maximum=9, actual=3)], policy, 0
    )
    request = result.requests[0]
    assert request.status == "completed"
    assert request.admitted_at_s == 0
    assert request.first_token_at_s == pytest.approx(0.4 + 1.5)
    assert request.completed_at_s == pytest.approx(0.4 + 3 * 1.5)
    assert request.revenue == 1.25 * 5
    assert result.prices[0].time_s == 0


def test_deadline_attempt_happens_before_expiry_and_equal_time_release():
    # The first order completes exactly when the second order's deadline arrives.
    server = ServerConfig(t0_s=1, k_s=0, prefill_s_per_token=0, max_batch_size=1)
    first = job("first", prompt=1, maximum=1, actual=1)
    second = job("second", arrival=0, prompt=1, maximum=1, actual=1, wait=1)

    class HoldFirst(AdmitAll):
        def step(self, pending, free, t):
            if t == 0:
                return Decision((Admission("first", 5),))
            return Decision((Admission("second", 5),)) if pending else Decision()

    result = Simulator(server).run([first, second], HoldFirst(), 0)
    assert [r.status for r in result.requests] == ["completed", "completed"]
    assert result.requests[1].admitted_at_s == pytest.approx(1)


def test_reservation_includes_full_max_output_until_completion():
    server = ServerConfig(t0_s=1, k_s=0, prefill_s_per_token=0, max_batch_size=3, max_kv_tokens=6)
    jobs = [job("a", prompt=1, maximum=4, actual=1), job("b", prompt=1, maximum=4, actual=1)]
    result = Simulator(server).run(jobs, AdmitAll(), 0)
    assert [r.status for r in result.requests] == ["completed", "timeout"]
    assert result.requests[1].admitted_at_s is None


def test_intrinsically_oversized_reservation_is_infeasible():
    result = Simulator(ServerConfig(max_kv_tokens=3)).run([job(prompt=2, maximum=2)], AdmitAll(), 0)
    assert result.requests[0].status == "infeasible"


def test_policy_only_sees_orders_and_price_is_locked_at_admission():
    policy = AdmitAll(rate=2)
    result = Simulator(ServerConfig(t0_s=0.25, k_s=0.1, prefill_s_per_token=0)).run(
        [job(actual=2)], policy, 0
    )
    assert all(isinstance(order, Order) for seen, _, _ in policy.seen for order in seen)
    assert result.requests[0].locked_rate_per_token == 2
    assert result.requests[0].revenue == 6


def test_bad_policy_rate_is_rejected():
    with pytest.raises(PolicyError):
        Simulator(ServerConfig()).run([job()], AdmitAll(rate=6), 0)


def test_arrivals_may_reserve_while_operation_is_busy():
    server = ServerConfig(t0_s=2, k_s=0, prefill_s_per_token=0, max_batch_size=2)
    policy = AdmitAll()
    result = Simulator(server).run([job("a"), job("b", arrival=0.5)], policy, 0.5)
    assert result.requests[1].admitted_at_s == pytest.approx(0.5)
    # The second request joins at the next decode cycle, after the first cycle ends.
    assert result.requests[1].first_token_at_s > result.requests[0].first_token_at_s


def test_admission_during_prefill_waits_for_fifo_prefill_cycles():
    server = ServerConfig(
        t0_s=1, k_s=0, prefill_s_per_token=1,
        prefill_chunk_tokens=1, max_batch_size=2,
    )
    result = Simulator(server).run(
        [job("long", prompt=2), job("arrives_during_prefill", arrival=0.5, prompt=1)],
        AdmitAll(), 0.5,
    )
    later = result.requests[1]
    assert later.admitted_at_s == pytest.approx(0.5)
    # A decode step runs between bounded prefill chunks once a job is ready.
    assert later.first_token_at_s == pytest.approx(5.0)


def test_horizon_is_inclusive_and_usage_extends_through_horizon():
    policy = AdmitAll()
    result = Simulator(ServerConfig(t0_s=1, k_s=0, prefill_s_per_token=0)).run(
        [job(arrival=1)], policy, 1
    )
    assert result.requests[0].status == "completed"
    assert result.elapsed_s >= 1
    assert result.usage[-1].end_s == result.elapsed_s


def test_ready_decode_is_not_blocked_by_an_entire_long_prefill():
    server = ServerConfig(t0_s=1, k_s=0, prefill_s_per_token=1,
                          prefill_chunk_tokens=1, max_batch_size=2)
    result = Simulator(server).run(
        [job("a-short", prompt=1, actual=2), job("b-long", prompt=4)], AdmitAll(), 0)
    assert result.requests[0].first_token_at_s == 2
    assert result.requests[0].completed_at_s == 4
    assert result.requests[1].first_token_at_s == 8


def test_new_quotes_do_not_change_existing_requests_rate():
    class ChangingQuote(AdmitAll):
        def step(self, pending, free, t):
            self.rate = 1 if t == 0 else 2
            return super().step(pending, free, t)

    server = ServerConfig(t0_s=1, k_s=0, prefill_s_per_token=0, max_batch_size=2)
    result = Simulator(server).run([job("a", actual=2), job("b", arrival=.5)],
                                   ChangingQuote(), .5)
    assert result.requests[0].locked_rate_per_token == 1
    assert result.requests[1].locked_rate_per_token == 2
    assert result.requests[0].revenue == 3
    assert result.requests[1].revenue == 4
