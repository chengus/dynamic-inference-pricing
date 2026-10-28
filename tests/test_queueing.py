"""Continuous-service validation isolated from the integer-token production model."""

import random

import pytest

from auction_sim.engine import Simulator
from auction_sim.models import Admission, Decision, Job, Order, ServerConfig


class FIFO:
    def step(self, pending, free_capacity, t):
        if pending and free_capacity.batch_slots:
            order = pending[0]
            return Decision((Admission(order.request_id, order.bid_per_token),))
        return Decision()

    def observe(self, interval):
        pass

    def next_wakeup(self, t):
        return None


def _busy_time_after(intervals, start):
    total = 0.0
    for interval in intervals:
        left = max(start, interval.start_s)
        right = interval.end_s
        if interval.busy and right > left:
            total += right - left
    return total


def test_seeded_mm1_waiting_and_utilization_match_analytic_values():
    # A narrowly scoped continuous-time service hook makes this a real M/M/1
    # check while leaving normal integer output-token timing untouched.
    rng = random.Random(20261027)
    arrival_rate, service_rate = 0.65, 1.0
    count, warmup_s = 8_000, 1_000.0
    jobs, service_times = [], {}
    arrival = 0.0
    for i in range(count):
        arrival += rng.expovariate(arrival_rate)
        rid = f"q{i}"
        order = Order(rid, arrival, 1, 1, 1.0, 1_000_000.0, "queue")
        jobs.append(Job(order, 1, 1.0))
        service_times[rid] = rng.expovariate(service_rate)

    server = ServerConfig(
        t0_s=1.0, k_s=0.0, prefill_s_per_token=0.0,
        max_batch_size=1, max_kv_tokens=2,
    )
    result = Simulator(server).run(
        jobs, FIFO(), arrival, service_times_s=service_times
    )

    waits = [
        request.admitted_at_s - request.job.order.arrival_s
        for request in result.requests
        if request.admitted_at_s is not None and request.job.order.arrival_s >= warmup_s
    ]
    observed_wait = sum(waits) / len(waits)
    observed_utilization = _busy_time_after(result.usage, warmup_s) / (result.elapsed_s - warmup_s)
    expected_wait = arrival_rate / (service_rate * (service_rate - arrival_rate))
    expected_utilization = arrival_rate / service_rate

    assert observed_wait == pytest.approx(expected_wait, abs=0.35)
    assert observed_utilization == pytest.approx(expected_utilization, abs=0.035)


def test_continuous_service_hook_rejects_non_mm1_workloads():
    server = ServerConfig(max_batch_size=2, prefill_s_per_token=0)
    order = Order("x", 0, 1, 1, 1, 10, "queue")
    job = Job(order, 1, 1)
    with pytest.raises(ValueError, match="max_batch_size=1"):
        Simulator(server).run([job], FIFO(), 0, service_times_s={"x": 1.0})
