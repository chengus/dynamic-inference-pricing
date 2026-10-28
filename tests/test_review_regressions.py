"""Review regressions that require no event-engine execution."""

import pytest

from auction_sim.baselines import FlatPrice, StaticTiers, TimeOfDayGrid
from auction_sim.mechanisms import UniformAuction
from auction_sim.models import FreeCapacity, Order


def test_no_online_policy_admits_an_order_before_its_arrival():
    future = Order("future", 10, 1, 1, 5.0, 0.0, "interactive")
    capacity = FreeCapacity(1, 10, 1, 10)
    policies = [UniformAuction(1), FlatPrice(1, 1), StaticTiers(1, 1, 1, 10),
                TimeOfDayGrid([1] * 24, 1)]
    for policy in policies:
        assert policy.step([future], capacity, 0).admitted == ()


def test_static_tier_waits_must_be_ordered():
    with pytest.raises(ValueError, match="standard_wait_s"):
        StaticTiers(1, 1, standard_wait_s=20, flex_wait_s=10)
