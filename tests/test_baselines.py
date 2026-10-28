import pytest

from auction_sim.baselines import FlatPrice, StaticTiers, TimeOfDayGrid, UtilizationController
from auction_sim.models import FreeCapacity, Order, UsageInterval


def order(
    request_id: str,
    bid: float,
    reserved: int = 2,
    arrival: float = 0,
    wait: float = 10,
) -> Order:
    return Order(request_id, arrival, 1, reserved - 1, bid, wait, "test")


def capacity(slots: int = 10, kv: int = 100, total_kv: int = 100) -> FreeCapacity:
    return FreeCapacity(slots, kv, 10, total_kv)


def test_flat_price_reserve_floor_and_fifo_blocking_prefix() -> None:
    policy = FlatPrice(price_per_token=2, reserve_per_token=3)
    result = policy.step(
        [order("first", 3, reserved=10), order("second", 3, reserved=9), order("third", 3, reserved=2)],
        capacity(kv=12),
        0,
    )
    assert [a.request_id for a in result.admitted] == ["first"]
    assert result.admitted[0].rate_per_token == 3
    assert result.clearing_price is None


def test_static_tiers_select_cheapest_affordable_wait_eligible_tier() -> None:
    policy = StaticTiers(10, 2, standard_wait_s=4, flex_wait_s=8)
    result = policy.step(
        [
            order("flex", 5, wait=8),
            order("standard", 10, wait=4),
            order("priority", 20, wait=0),
            order("unavailable", 15, wait=3),
        ],
        capacity(),
        0,
    )
    assert [(a.request_id, a.tier, a.rate_per_token) for a in result.admitted] == [
        ("priority", "priority", 20),
        ("standard", "standard", 10),
        ("flex", "flex", 5),
    ]


def test_tier_schedule_stops_at_first_blocking_order() -> None:
    policy = StaticTiers(10, 2, standard_wait_s=4, flex_wait_s=8)
    result = policy.step(
        [order("flex", 5, reserved=2, wait=8),
         order("priority-blocker", 20, reserved=3, wait=0),
         order("standard", 10, wait=4)],
        capacity(kv=2),
        0,
    )
    assert result.admitted == ()


def test_time_grid_hourly_quotes_and_wakeup_boundaries() -> None:
    policy = TimeOfDayGrid([10] * 23 + [2], reserve_per_token=3)
    assert policy.quote_at(0) == 10
    assert policy.quote_at(23 * 3600) == 3
    assert policy.quote_at(24 * 3600) == 10
    assert policy.next_wakeup(0) == 3600
    assert policy.next_wakeup(3600) == 7200
    assert policy.next_wakeup(24 * 3600) == 25 * 3600

    result = policy.step([order("waiting", 4, arrival=23 * 3600)], capacity(), 23 * 3600)
    assert [a.request_id for a in result.admitted] == ["waiting"]
    assert result.admitted[0].rate_per_token == 3


def test_time_grid_custom_edges_wake_when_price_changes() -> None:
    policy = TimeOfDayGrid([10, 2, 8], 1, edges_s=[0, 100, 1000, 86400])
    assert policy.quote_at(99.999) == 10
    assert policy.quote_at(100) == 2
    assert policy.quote_at(1000) == 8
    assert policy.next_wakeup(500) == 1000
    assert policy.next_wakeup(1000) == 86400
    assert policy.next_wakeup(86400) == 86500


def test_utilization_controller_updates_on_real_time_window_and_clamps() -> None:
    policy = UtilizationController(
        initial_price_per_token=10,
        reserve_per_token=2,
        target_utilization=0.5,
        update_interval_s=10,
        gain=1,
        min_price=9,
        max_price=12,
    )
    assert policy.next_wakeup(0) == 10
    policy.observe(UsageInterval(0, 8, busy=True, active_requests=1, decoding_requests=1, reserved_kv_tokens=1))
    # The two idle seconds are part of the ten-second window.
    policy.observe(UsageInterval(8, 10, busy=False, active_requests=0, decoding_requests=0, reserved_kv_tokens=0))
    assert policy.step([], capacity(), 10).admitted == ()
    assert policy.price_per_token == 12
    assert policy.next_wakeup(10) == 20

    policy.observe(UsageInterval(10, 20, busy=False, active_requests=0, decoding_requests=0, reserved_kv_tokens=0))
    policy.step([], capacity(), 20)
    assert policy.price_per_token == pytest.approx(9)


def test_utilization_quote_admits_at_current_price_only() -> None:
    policy = UtilizationController(5, 1, update_interval_s=10)
    result = policy.step([order("affordable", 5), order("not-affordable", 4)], capacity(), 0)
    assert [a.request_id for a in result.admitted] == ["affordable"]
    assert result.admitted[0].rate_per_token == 5
