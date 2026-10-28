from auction_sim.mechanisms import FirstPriceHybridAuction, UniformAuction
from auction_sim.models import FreeCapacity, Order


def order(
    request_id: str,
    bid: float,
    reserved: int = 2,
    arrival: float = 0,
    wait: float = 10,
    order_type: str = "limit",
) -> Order:
    return Order(
        request_id=request_id,
        arrival_s=arrival,
        prompt_tokens=1,
        max_tokens=reserved - 1,
        bid_per_token=bid,
        max_wait_s=wait,
        segment="test",
        order_type=order_type,  # type: ignore[arg-type]
    )


def capacity(slots: int = 10, kv: int = 100, total_kv: int = 100) -> FreeCapacity:
    return FreeCapacity(slots, kv, 10, total_kv)


def test_auction_admits_prefix_and_does_not_skip_blocked_order() -> None:
    auction = UniformAuction(reserve_per_token=1)
    result = auction.step(
        [order("a", 10, 10), order("b", 9, 9), order("c", 2, 2)],
        capacity(kv=12),
        t=0,
    )

    assert [admission.request_id for admission in result.admitted] == ["a"]
    assert result.clearing_price == 9
    assert result.admitted[0].rate_per_token == 9


def test_auction_filters_expired_infeasible_and_below_reserve() -> None:
    auction = UniformAuction(reserve_per_token=3)
    result = auction.step(
        [
            order("expired", 20, wait=1),
            order("too-large", 19, reserved=21),
            order("below", 2),
            order("boundary", 3, arrival=2, wait=0),
        ],
        capacity(kv=50, total_kv=20),
        t=2,
    )
    assert [admission.request_id for admission in result.admitted] == ["boundary"]
    assert result.clearing_price == 3


def test_auction_deadline_equality_is_eligible_and_empty_admission_has_no_price() -> None:
    auction = UniformAuction(reserve_per_token=1)
    assert auction.step([order("at-deadline", 1, wait=2)], capacity(), 2).admitted
    result = auction.step([order("no-capacity", 1)], capacity(slots=0), 0)
    assert result.admitted == ()
    assert result.clearing_price is None


def test_auction_ties_use_arrival_then_request_id() -> None:
    auction = UniformAuction(reserve_per_token=1)
    result = auction.step(
        [order("z", 5, arrival=0), order("b", 5, arrival=1), order("a", 5, arrival=0)],
        capacity(slots=2),
        1,
    )
    assert [admission.request_id for admission in result.admitted] == ["a", "z"]
    assert result.clearing_price == 5


def test_hybrid_charges_limits_first_price_and_market_clearing_price() -> None:
    auction = FirstPriceHybridAuction(reserve_per_token=1)
    result = auction.step(
        [order("market", 10, order_type="market"), order("limit", 8)],
        capacity(slots=1),
        0,
    )
    assert [a.request_id for a in result.admitted] == ["market"]
    assert result.admitted[0].rate_per_token == 8
    assert result.clearing_price == 8

    result = auction.step([order("limit", 8)], capacity(), 0)
    assert result.admitted[0].rate_per_token == 8
    assert result.clearing_price == 1
