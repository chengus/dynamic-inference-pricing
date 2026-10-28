from dataclasses import FrozenInstanceError

import pytest

from auction_sim.models import Job, Order, ServerConfig


def order(**overrides):
    return Order(**dict(request_id="a", arrival_s=0.0, prompt_tokens=4,
                        max_tokens=8, bid_per_token=2.0, max_wait_s=0.0,
                        segment="interactive", **overrides))


def test_public_order_excludes_private_truth_and_is_immutable():
    public = order()
    assert not hasattr(public, "actual_output_tokens")
    assert not hasattr(public, "value_per_token")
    assert public.reserved_kv_tokens == 12
    with pytest.raises(FrozenInstanceError):
        public.max_tokens = 16


def test_billable_tokens_use_actual_output_not_reserved_output():
    private = Job(order(), actual_output_tokens=3, value_per_token=5.0)
    assert private.billable_tokens == 7
    assert private.total_value == 35.0


@pytest.mark.parametrize("field,value", [
    ("arrival_s", -1), ("arrival_s", float("inf")), ("bid_per_token", float("nan")),
    ("max_wait_s", True), ("prompt_tokens", 0), ("max_tokens", 1.5),
    ("order_type", "unknown"),
])
def test_invalid_public_fields_rejected(field, value):
    values = dict(request_id="a", arrival_s=0.0, prompt_tokens=4, max_tokens=8,
                  bid_per_token=2.0, max_wait_s=0.0, segment="interactive")
    values[field] = value
    with pytest.raises(ValueError):
        Order(**values)


@pytest.mark.parametrize("actual", [0, 9, 1.5, True])
def test_realized_output_must_be_bounded_integer(actual):
    with pytest.raises(ValueError):
        Job(order(), actual, 1.0)


def test_decode_progress_requires_positive_duration():
    with pytest.raises(ValueError):
        ServerConfig(t0_s=0, k_s=0)
