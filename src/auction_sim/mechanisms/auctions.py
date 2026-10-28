"""Batch auctions for public order fields."""

from __future__ import annotations

import math
from collections.abc import Sequence

from auction_sim.models import Admission, Decision, FreeCapacity, Order, UsageInterval


def _positive_rate(name: str, value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


class _BatchAuction:
    def __init__(self, reserve_per_token: float) -> None:
        self.reserve_per_token = _positive_rate("reserve_per_token", reserve_per_token)

    def _winner_rate(self, order: Order, clearing_price: float) -> float:
        raise NotImplementedError

    def step(self, pending: Sequence[Order], free_capacity: FreeCapacity, t: float) -> Decision:
        eligible = [
            order
            for order in pending
            if order.arrival_s <= t <= order.deadline_s
            and order.reserved_kv_tokens <= free_capacity.max_kv_tokens
            and order.bid_per_token >= self.reserve_per_token
        ]
        eligible.sort(key=lambda order: (-order.bid_per_token, order.arrival_s, order.request_id))
        winners: list[Order] = []
        slots_left = max(0, free_capacity.batch_slots)
        kv_left = max(0, free_capacity.kv_tokens)
        for order in eligible:
            if slots_left < 1 or order.reserved_kv_tokens > kv_left:
                break
            winners.append(order)
            slots_left -= 1
            kv_left -= order.reserved_kv_tokens

        if not winners:
            return Decision()

        # The next eligible bid is the marginal losing bid, including a request
        # blocked by the remaining batch or KV capacity.
        highest_losing_bid = (
            eligible[len(winners)].bid_per_token
            if len(winners) < len(eligible)
            else self.reserve_per_token
        )
        clearing_price = max(self.reserve_per_token, highest_losing_bid)
        admitted = tuple(
            Admission(order.request_id, self._winner_rate(order, clearing_price))
            for order in winners
        )
        return Decision(admitted=admitted, clearing_price=clearing_price)

    def observe(self, interval: UsageInterval) -> None:
        return None

    def next_wakeup(self, t: float) -> float | None:
        return None


class UniformAuction(_BatchAuction):
    """Admit a ranked prefix and charge every winner the clearing price."""

    def _winner_rate(self, order: Order, clearing_price: float) -> float:
        return clearing_price


class FirstPriceHybridAuction(_BatchAuction):
    """Limit orders pay their bid; market orders pay the batch clearing price."""

    def _winner_rate(self, order: Order, clearing_price: float) -> float:
        if order.order_type == "market":
            return clearing_price
        return order.bid_per_token
