"""Posted-price baselines using only public order and capacity data."""

from __future__ import annotations

import math
from collections.abc import Sequence

from auction_sim.models import Admission, Decision, FreeCapacity, Order, UsageInterval

_DAY_S = 24 * 60 * 60


def _finite_nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _affordable_fifo(
    pending: Sequence[Order], free_capacity: FreeCapacity, t: float, price: float
) -> tuple[Order, ...]:
    eligible = [
        order
        for order in pending
        if order.arrival_s <= t <= order.deadline_s
        and order.reserved_kv_tokens <= free_capacity.max_kv_tokens
        and order.bid_per_token >= price
    ]
    eligible.sort(key=lambda order: (order.arrival_s, order.request_id))
    slots_left = max(0, free_capacity.batch_slots)
    kv_left = max(0, free_capacity.kv_tokens)
    winners: list[Order] = []
    for order in eligible:
        if slots_left < 1 or order.reserved_kv_tokens > kv_left:
            break
        winners.append(order)
        slots_left -= 1
        kv_left -= order.reserved_kv_tokens
    return tuple(winners)


class _StaticPolicy:
    def observe(self, interval: UsageInterval) -> None:
        return None

    def next_wakeup(self, t: float) -> float | None:
        return None


class FlatPrice(_StaticPolicy):
    def __init__(self, price_per_token: float, reserve_per_token: float) -> None:
        self.price_per_token = _finite_nonnegative("price_per_token", price_per_token)
        self.reserve_per_token = _positive("reserve_per_token", reserve_per_token)

    @property
    def quote(self) -> float:
        return max(self.price_per_token, self.reserve_per_token)

    def step(self, pending: Sequence[Order], free_capacity: FreeCapacity, t: float) -> Decision:
        admitted = tuple(
            Admission(order.request_id, self.quote)
            for order in _affordable_fifo(pending, free_capacity, t, self.quote)
        )
        return Decision(admitted=admitted)


class StaticTiers(_StaticPolicy):
    """Flex, standard, and priority offers at 0.5x, 1x, and 2x base price."""

    _TIER_ORDER = {"priority": 0, "standard": 1, "flex": 2}
    _MULTIPLIERS = {"flex": 0.5, "standard": 1.0, "priority": 2.0}

    def __init__(
        self,
        base_price_per_token: float,
        reserve_per_token: float,
        standard_wait_s: float,
        flex_wait_s: float,
    ) -> None:
        self.base_price_per_token = _finite_nonnegative("base_price_per_token", base_price_per_token)
        self.reserve_per_token = _positive("reserve_per_token", reserve_per_token)
        self.standard_wait_s = _finite_nonnegative("standard_wait_s", standard_wait_s)
        self.flex_wait_s = _finite_nonnegative("flex_wait_s", flex_wait_s)
        if self.standard_wait_s > self.flex_wait_s:
            raise ValueError("standard_wait_s must not exceed flex_wait_s")
        self._prices = {
            name: max(self.reserve_per_token, self.base_price_per_token * multiplier)
            for name, multiplier in self._MULTIPLIERS.items()
        }

    def _select_tier(self, order: Order) -> str | None:
        advertised_wait = {
            "priority": 0.0,
            "standard": self.standard_wait_s,
            "flex": self.flex_wait_s,
        }
        options = [
            tier
            for tier, price in self._prices.items()
            if advertised_wait[tier] <= order.max_wait_s and order.bid_per_token >= price
        ]
        if not options:
            return None
        return min(options, key=lambda tier: (self._prices[tier], self._TIER_ORDER[tier]))

    def step(self, pending: Sequence[Order], free_capacity: FreeCapacity, t: float) -> Decision:
        selected: list[tuple[Order, str]] = []
        for order in pending:
            if (
                not order.arrival_s <= t <= order.deadline_s
                or order.reserved_kv_tokens > free_capacity.max_kv_tokens
            ):
                continue
            tier = self._select_tier(order)
            if tier is not None:
                selected.append((order, tier))
        selected.sort(key=lambda pair: (self._TIER_ORDER[pair[1]], pair[0].arrival_s, pair[0].request_id))
        slots_left = max(0, free_capacity.batch_slots)
        kv_left = max(0, free_capacity.kv_tokens)
        admitted: list[Admission] = []
        for order, tier in selected:
            if slots_left < 1 or order.reserved_kv_tokens > kv_left:
                break
            admitted.append(Admission(order.request_id, self._prices[tier], tier))
            slots_left -= 1
            kv_left -= order.reserved_kv_tokens
        return Decision(admitted=tuple(admitted))


class TimeOfDayGrid(_StaticPolicy):
    """Daily posted-price grid.

    ``prices_per_token`` contains 24 hourly prices by default. For other slot
    widths, supply ``edges_s`` as increasing day-second boundaries beginning
    at 0 and ending at 86400; there must be one more edge than prices.
    """

    def __init__(
        self,
        prices_per_token: Sequence[float],
        reserve_per_token: float,
        edges_s: Sequence[float] | None = None,
    ) -> None:
        if not prices_per_token:
            raise ValueError("prices_per_token must be nonempty")
        self.prices_per_token = tuple(
            _finite_nonnegative("prices_per_token entry", price) for price in prices_per_token
        )
        self.reserve_per_token = _positive("reserve_per_token", reserve_per_token)
        if edges_s is None:
            if len(self.prices_per_token) != 24:
                raise ValueError("exactly 24 prices are required when edges_s is omitted")
            self.edges_s = tuple(3600.0 * hour for hour in range(25))
        else:
            edges = tuple(_finite_nonnegative("edges_s entry", edge) for edge in edges_s)
            if len(edges) != len(self.prices_per_token) + 1:
                raise ValueError("edges_s must contain one more entry than prices_per_token")
            if edges[0] != 0 or edges[-1] != _DAY_S:
                raise ValueError("edges_s must start at 0 and end at 86400")
            if any(right <= left for left, right in zip(edges, edges[1:])):
                raise ValueError("edges_s must be strictly increasing")
            self.edges_s = edges

    def _slot_at(self, t: float) -> int:
        day_time = t % _DAY_S
        # Day boundaries use the first slot; internal boundaries enter the slot
        # on their right, matching half-open price intervals.
        for i, edge in enumerate(self.edges_s[1:]):
            if day_time < edge:
                return i
        return 0

    def quote_at(self, t: float) -> float:
        return max(self.reserve_per_token, self.prices_per_token[self._slot_at(t)])

    def step(self, pending: Sequence[Order], free_capacity: FreeCapacity, t: float) -> Decision:
        quote = self.quote_at(t)
        admitted = tuple(
            Admission(order.request_id, quote)
            for order in _affordable_fifo(pending, free_capacity, t, quote)
        )
        return Decision(admitted=admitted)

    def next_wakeup(self, t: float) -> float | None:
        if not math.isfinite(t):
            raise ValueError("t must be finite")
        day_start = math.floor(t / _DAY_S) * _DAY_S
        day_time = t - day_start
        for edge in self.edges_s[1:-1]:
            if edge > day_time:
                return day_start + edge
        return day_start + _DAY_S


class UtilizationController:
    """Periodic posted-price controller using observed busy-time fractions.

    Its minimum quote is floored at the reserve. The reserve is a configured
    price floor, not a claim that the quote covers operating cost.
    """

    def __init__(
        self,
        initial_price_per_token: float,
        reserve_per_token: float,
        target_utilization: float = 0.8,
        update_interval_s: float = 60.0,
        gain: float = 0.1,
        min_price: float | None = None,
        max_price: float | None = None,
    ) -> None:
        self.reserve_per_token = _positive("reserve_per_token", reserve_per_token)
        self.target_utilization = _positive("target_utilization", target_utilization)
        if self.target_utilization >= 1:
            raise ValueError("target_utilization must be less than 1")
        self.update_interval_s = _positive("update_interval_s", update_interval_s)
        self.gain = _finite_nonnegative("gain", gain)
        floor = self.reserve_per_token if min_price is None else _finite_nonnegative("min_price", min_price)
        self.min_price = max(self.reserve_per_token, floor)
        self.max_price = None if max_price is None else _positive("max_price", max_price)
        if self.max_price is not None and self.max_price < self.min_price:
            raise ValueError("max_price must be at least max(reserve_per_token, min_price)")
        initial = _positive("initial_price_per_token", initial_price_per_token)
        self.price_per_token = self._clamp(initial)
        self._busy_s = 0.0
        self._next_update_s = self.update_interval_s

    def _clamp(self, price: float) -> float:
        price = max(price, self.min_price)
        if self.max_price is not None:
            price = min(price, self.max_price)
        return price

    def observe(self, interval: UsageInterval) -> None:
        duration = interval.end_s - interval.start_s
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("usage interval must have finite, nonnegative duration")
        if interval.busy:
            self._busy_s += duration

    def _update_through(self, t: float) -> None:
        while t >= self._next_update_s:
            utilization = min(1.0, max(0.0, self._busy_s / self.update_interval_s))
            exponent = self.gain * (utilization - self.target_utilization)
            # Work in log space to avoid overflow on a long run without a max.
            log_price = math.log(self.price_per_token) + exponent
            log_price = max(log_price, math.log(self.min_price))
            if self.max_price is not None:
                log_price = min(log_price, math.log(self.max_price))
            # Bound even an uncapped controller to finite representable rates.
            log_price = min(log_price, math.log(float.fromhex("0x1.fffffffffffffp+1023")))
            self.price_per_token = self._clamp(math.exp(log_price))
            self._busy_s = 0.0
            self._next_update_s += self.update_interval_s

    def step(self, pending: Sequence[Order], free_capacity: FreeCapacity, t: float) -> Decision:
        if not math.isfinite(t):
            raise ValueError("t must be finite")
        self._update_through(t)
        quote = self.price_per_token
        admitted = tuple(
            Admission(order.request_id, quote)
            for order in _affordable_fifo(pending, free_capacity, t, quote)
        )
        return Decision(admitted=admitted)

    def next_wakeup(self, t: float) -> float | None:
        if not math.isfinite(t):
            raise ValueError("t must be finite")
        if self._next_update_s > t:
            return self._next_update_s
        elapsed_steps = math.floor((t - self._next_update_s) / self.update_interval_s) + 1
        return self._next_update_s + elapsed_steps * self.update_interval_s
