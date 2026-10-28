"""Discrete event engine for the admission auction simulator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Mapping

from ..models import (
    ActiveRequestSnapshot, Admission, Decision, FreeCapacity, Job, Policy, PricePoint,
    RequestResult, ServerConfig, SimulationResult, SimulationSnapshot, UsageInterval,
)


class PolicyError(ValueError):
    """Raised when a policy returns an invalid decision."""


@dataclass
class _Live:
    job: Job
    admitted: float
    rate: float
    tier: str | None
    prompt_left: int
    output_done: int = 0
    first_token: float | None = None
    prefill_ready: bool = False


class Simulator:
    """Run jobs on a server using an online admission policy."""

    def __init__(self, server: ServerConfig):
        self.server = server

    def run(
        self, jobs: Iterable[Job], policy: Policy, arrival_horizon_s: float,
        *, service_times_s: Mapping[str, float] | None = None,
        observer: Callable[[SimulationSnapshot], None] | None = None,
    ) -> SimulationResult:
        jobs = tuple(jobs)
        if not math.isfinite(arrival_horizon_s) or arrival_horizon_s < 0:
            raise ValueError("arrival_horizon_s must be finite and nonnegative")
        ids: set[str] = set()
        for job in jobs:
            if job.order.request_id in ids:
                raise ValueError(f"duplicate request_id: {job.order.request_id}")
            ids.add(job.order.request_id)
            if job.order.arrival_s > arrival_horizon_s:
                raise ValueError("all arrivals must lie in [0, arrival_horizon_s]")
        if service_times_s is not None:
            if self.server.max_batch_size != 1 or self.server.prefill_s_per_token != 0:
                raise ValueError("service_times_s requires max_batch_size=1 and zero prefill time")
            if any(job.actual_output_tokens != 1 for job in jobs):
                raise ValueError("service_times_s requires one output token per job")
            known_ids = {job.order.request_id for job in jobs}
            if set(service_times_s) - known_ids:
                raise ValueError("service_times_s contains an unknown request_id")
            if any(not math.isfinite(value) or value <= 0 for value in service_times_s.values()):
                raise ValueError("service times must be finite and positive")

        ordered = sorted(jobs, key=lambda j: (j.order.arrival_s, j.order.request_id))
        pending: dict[str, Job] = {}
        live: dict[str, _Live] = {}
        results: dict[str, RequestResult] = {}
        prefill_fifo: list[str] = []
        price_points: list[PricePoint] = []
        usage: list[UsageInterval] = []
        arrival_i = 0
        t = 0.0
        last_t = 0.0
        op_end: float | None = None
        op_kind: str | None = None
        op_plan: dict[str, int] = {}
        next_phase = "prefill"
        policy_wakeup: float | None = None
        last_decision_t: float | None = None
        interval_busy = False
        interval_active = interval_decoding = interval_kv = 0
        completed_count = timeout_count = infeasible_count = admitted_count = 0
        revenue_observed = 0.0

        def capacity() -> FreeCapacity:
            return FreeCapacity(
                self.server.max_batch_size - len(live),
                self.server.max_kv_tokens - sum(x.job.order.reserved_kv_tokens for x in live.values()),
                self.server.max_batch_size, self.server.max_kv_tokens,
            )

        def record(status: str, job: Job, *, item: _Live | None = None, at: float | None = None) -> None:
            nonlocal completed_count, timeout_count, infeasible_count, revenue_observed
            results[job.order.request_id] = RequestResult(
                job=job, status=status,
                admitted_at_s=item.admitted if item else None,
                first_token_at_s=item.first_token if item else None,
                completed_at_s=at if status == "completed" else None,
                locked_rate_per_token=item.rate if item else None,
                tier=item.tier if item else None,
            )
            if status == "completed":
                completed_count += 1
                revenue_observed += results[job.order.request_id].revenue
            elif status == "timeout":
                timeout_count += 1
            elif status == "infeasible":
                infeasible_count += 1

        def publish_snapshot(event: str, now: float) -> None:
            if observer is None:
                return
            active = tuple(
                ActiveRequestSnapshot(
                    request_id=rid, admitted_at_s=item.admitted,
                    rate_per_token=item.rate, tier=item.tier,
                    prompt_tokens=item.job.order.prompt_tokens,
                    max_tokens=item.job.order.max_tokens,
                    segment=item.job.order.segment,
                    order_type=item.job.order.order_type,
                )
                for rid, item in live.items()
            )
            observer(SimulationSnapshot(
                time_s=now, event=event,
                pending=tuple(job.order for job in pending.values()),
                active=active, free_capacity=capacity(), busy=op_end is not None,
                completed_count=completed_count, timeout_count=timeout_count,
                infeasible_count=infeasible_count, admitted_count=admitted_count,
                revenue_observed=revenue_observed,
                latest_clearing_price=price_points[-1] if price_points else None,
            ))

        def settle_operation(now: float) -> None:
            nonlocal op_end, op_kind, op_plan, prefill_fifo
            if op_end is None or op_end > now:
                return
            if op_kind == "prefill":
                for rid, tokens in op_plan.items():
                    item = live.get(rid)
                    if item is None:
                        continue
                    item.prompt_left -= tokens
                    if item.prompt_left == 0:
                        item.prefill_ready = True
                        if rid in prefill_fifo:
                            prefill_fifo.remove(rid)
            elif op_kind == "decode":
                for rid in op_plan:
                    item = live.get(rid)
                    if item is None:
                        continue
                    if item.first_token is None:
                        item.first_token = now
                    item.output_done += 1
                    if item.output_done >= item.job.actual_output_tokens:
                        record("completed", item.job, item=item, at=now)
                        del live[rid]
            op_end = None
            op_kind = None
            op_plan = {}

        def schedule_operation(now: float) -> None:
            nonlocal op_end, op_kind, op_plan, prefill_fifo, next_phase
            if op_end is not None:
                return
            ready = [rid for rid, item in live.items() if item.prefill_ready and item.output_done < item.job.actual_output_tokens]
            for phase in (next_phase, "decode" if next_phase == "prefill" else "prefill"):
                if phase == "prefill":
                    budget = self.server.prefill_chunk_tokens
                    plan: dict[str, int] = {}
                    for rid in prefill_fifo:
                        item = live.get(rid)
                        if item is None or item.prompt_left <= 0:
                            continue
                        amount = min(item.prompt_left, budget)
                        plan[rid] = amount
                        budget -= amount
                        if budget == 0:
                            break
                    if plan:
                        tokens = sum(plan.values())
                        op_kind, op_plan = "prefill", plan
                        op_end = now + tokens * self.server.prefill_s_per_token
                        next_phase = "decode"
                        return
                elif ready:
                    op_kind, op_plan = "decode", {rid: 1 for rid in ready}
                    if service_times_s is not None and ready[0] in service_times_s:
                        duration = service_times_s[ready[0]]
                    else:
                        duration = self.server.t0_s + self.server.k_s * len(ready)
                    op_end = now + duration
                    next_phase = "prefill"
                    return

        publish_snapshot("initial", 0.0)
        while len(results) < len(jobs):
            next_arrival = ordered[arrival_i].order.arrival_s if arrival_i < len(ordered) else None
            next_deadline = min((job.order.deadline_s for job in pending.values()), default=None)
            wakeup = policy_wakeup
            candidates = [x for x in (next_arrival, next_deadline, op_end, wakeup) if x is not None and x >= t]
            if not candidates:
                # A pending job must eventually hit a deadline; a live job must have an operation.
                if live:
                    schedule_operation(t)
                    if op_end is not None:
                        continue
                raise RuntimeError("simulator reached a state with no future event")
            t = min(candidates)
            if t < last_t:
                raise RuntimeError("event time moved backwards")

            if t > last_t:
                usage.append(UsageInterval(last_t, t, interval_busy, interval_active, interval_decoding, interval_kv))
                policy.observe(usage[-1])
            operation_boundary = op_end is not None and op_end <= t
            settle_operation(t)
            arrived = False
            while arrival_i < len(ordered) and ordered[arrival_i].order.arrival_s == t:
                job = ordered[arrival_i]
                arrival_i += 1
                arrived = True
                order = job.order
                if order.reserved_kv_tokens > self.server.max_kv_tokens or self.server.max_batch_size < 1:
                    record("infeasible", job)
                else:
                    pending[order.request_id] = job

            # One policy call per timestamp, covering arrivals, releases, deadlines, and wakeups.
            admitted_now = 0
            event_here = (
                (last_decision_t != t)
                and (t == next_arrival or t == next_deadline or t == wakeup or
                     operation_boundary or
                     (arrival_i > 0 and ordered[arrival_i - 1].order.arrival_s == t))
            )
            # `settle_operation` clears op_end, so release is identified by an operation boundary.
            # The event condition below also handles a completion when there were no arrivals.
            if last_decision_t != t and (event_here or t == last_t):
                free = capacity()
                snapshot = tuple(job.order for job in pending.values())
                decision = policy.step(snapshot, free, t)
                if not isinstance(decision, Decision):
                    raise PolicyError("step must return Decision")
                _apply_decision(decision, pending, live, prefill_fifo, price_points, t, free, self.server)
                admitted_now = len(decision.admitted)
                admitted_count += admitted_now
                last_decision_t = t
                policy_wakeup = policy.next_wakeup(t)
                if policy_wakeup is not None:
                    if not math.isfinite(policy_wakeup) or policy_wakeup <= t:
                        raise PolicyError("next_wakeup must be finite and strictly greater than t")
            # Deadline expiry occurs only after the admission attempt at that timestamp.
            expired = [rid for rid, job in pending.items() if job.order.deadline_s <= t]
            for rid in expired:
                record("timeout", pending.pop(rid))
            schedule_operation(t)

            interval_busy = op_end is not None
            interval_active = len(live)
            interval_decoding = len(op_plan) if op_kind == "decode" else 0
            interval_kv = sum(item.job.order.reserved_kv_tokens for item in live.values())
            last_t = t

            event_parts = []
            if arrived:
                event_parts.append("arrival")
            if operation_boundary:
                event_parts.append("release")
            if wakeup is not None and wakeup == t:
                event_parts.append("policy_wakeup")
            if expired:
                event_parts.append("expiry")
            if admitted_now:
                event_parts.append("admission")
            publish_snapshot("+".join(event_parts) or "event", t)

        # All requests are terminal. Include the final service interval through completion.
        elapsed = max(arrival_horizon_s, last_t)
        if elapsed > last_t:
            idle = UsageInterval(last_t, elapsed, False, 0, 0, 0)
            usage.append(idle)
            policy.observe(idle)
        publish_snapshot("final", elapsed)
        return SimulationResult(
            requests=tuple(results[j.order.request_id] for j in jobs),
            usage=tuple(usage), prices=tuple(price_points),
            arrival_horizon_s=arrival_horizon_s, elapsed_s=elapsed,
        )


def _apply_decision(decision, pending, live, prefill_fifo, prices, t, free, server):
    ids: set[str] = set()
    admitted_count = 0
    total_rate = None
    for admission in decision.admitted:
        if not isinstance(admission, Admission):
            raise PolicyError("admitted entries must be Admission")
        rid = admission.request_id
        if rid in ids:
            raise PolicyError("a decision may not admit a request twice")
        ids.add(rid)
        if rid not in pending:
            raise PolicyError(f"request {rid!r} is not pending")
        if not math.isfinite(admission.rate_per_token) or admission.rate_per_token < 0:
            raise PolicyError("admission rates must be finite and nonnegative")
        order = pending[rid].order
        if admission.rate_per_token > order.bid_per_token:
            raise PolicyError("admission rate exceeds the order bid")
    if len(ids) > free.batch_slots:
        raise PolicyError("decision exceeds available batch slots")
    if sum(pending[rid].order.reserved_kv_tokens for rid in ids) > free.kv_tokens:
        raise PolicyError("decision exceeds available KV capacity")
    for admission in decision.admitted:
        job = pending.pop(admission.request_id)
        live[admission.request_id] = _Live(job, t, admission.rate_per_token, admission.tier, job.order.prompt_tokens)
        prefill_fifo.append(admission.request_id)
        admitted_count += 1
        total_rate = (total_rate or 0.0) + admission.rate_per_token
    if admitted_count:
        price = decision.clearing_price
        kind = "clearing"
        if price is None:
            price = total_rate / admitted_count if admitted_count else 0.0
            kind = "mean_admission"
        if not math.isfinite(price) or price < 0:
            raise PolicyError("clearing_price must be finite and nonnegative")
        prices.append(PricePoint(t, price, admitted_count, kind))


def simulate(jobs, policy, server: ServerConfig, arrival_horizon_s: float, *, service_times_s=None) -> SimulationResult:
    return Simulator(server).run(jobs, policy, arrival_horizon_s, service_times_s=service_times_s)
