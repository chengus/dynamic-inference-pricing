"""Shared public contracts; policies never receive realized lengths or values."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Protocol, Sequence


@dataclass(frozen=True)
class Order:
    request_id: str
    arrival_s: float
    prompt_tokens: int
    max_tokens: int
    bid_per_token: float
    max_wait_s: float
    segment: str
    order_type: Literal["limit", "market"] = "limit"

    def __post_init__(self) -> None:
        if not self.request_id or not self.segment:
            raise ValueError("request_id and segment must be nonempty")
        for name in ("arrival_s", "bid_per_token", "max_wait_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.arrival_s + self.max_wait_s):
            raise ValueError("admission deadline must be finite")
        for name in ("prompt_tokens", "max_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.order_type not in ("limit", "market"):
            raise ValueError("order_type must be limit or market")

    @property
    def deadline_s(self) -> float:
        return self.arrival_s + self.max_wait_s

    @property
    def reserved_kv_tokens(self) -> int:
        return self.prompt_tokens + self.max_tokens


@dataclass(frozen=True)
class Job:
    """Simulator-only truth. Never pass Job objects to an online policy."""

    order: Order
    actual_output_tokens: int
    value_per_token: float

    def __post_init__(self) -> None:
        if not isinstance(self.actual_output_tokens, int) or isinstance(self.actual_output_tokens, bool):
            raise ValueError("actual_output_tokens must be an integer")
        if not 1 <= self.actual_output_tokens <= self.order.max_tokens:
            raise ValueError("actual output must lie in [1, max_tokens]")
        if isinstance(self.value_per_token, bool) or not math.isfinite(self.value_per_token) or self.value_per_token < 0:
            raise ValueError("value_per_token must be finite and nonnegative")

    @property
    def billable_tokens(self) -> int:
        return self.order.prompt_tokens + self.actual_output_tokens

    @property
    def total_value(self) -> float:
        return self.value_per_token * self.billable_tokens


@dataclass(frozen=True)
class ServerConfig:
    """Synthetic defaults, not a hardware benchmark."""

    t0_s: float = 0.005
    k_s: float = 0.0005
    prefill_s_per_token: float = 0.00002
    prefill_chunk_tokens: int = 256
    max_batch_size: int = 32
    max_kv_tokens: int = 65536

    def __post_init__(self) -> None:
        for name in ("t0_s", "k_s", "prefill_s_per_token"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.t0_s + self.k_s <= 0:
            raise ValueError("decode step duration must be positive")
        for name in ("prefill_chunk_tokens", "max_batch_size", "max_kv_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class FreeCapacity:
    batch_slots: int
    kv_tokens: int
    max_batch_size: int
    max_kv_tokens: int


@dataclass(frozen=True)
class Admission:
    request_id: str
    rate_per_token: float
    tier: str | None = None


@dataclass(frozen=True)
class Decision:
    admitted: tuple[Admission, ...] = ()
    clearing_price: float | None = None


@dataclass(frozen=True)
class UsageInterval:
    start_s: float
    end_s: float
    busy: bool
    active_requests: int
    decoding_requests: int
    reserved_kv_tokens: int


class Policy(Protocol):
    def step(self, pending: Sequence[Order], free_capacity: FreeCapacity, t: float) -> Decision: ...
    def observe(self, interval: UsageInterval) -> None: ...
    def next_wakeup(self, t: float) -> float | None: ...


@dataclass(frozen=True)
class RequestResult:
    job: Job
    status: Literal["completed", "timeout", "infeasible"]
    admitted_at_s: float | None = None
    first_token_at_s: float | None = None
    completed_at_s: float | None = None
    locked_rate_per_token: float | None = None
    tier: str | None = None

    @property
    def revenue(self) -> float:
        if self.status != "completed" or self.locked_rate_per_token is None:
            return 0.0
        return self.locked_rate_per_token * self.job.billable_tokens


@dataclass(frozen=True)
class PricePoint:
    time_s: float
    price_per_token: float
    admissions: int
    kind: Literal["clearing", "mean_admission"] = "clearing"


@dataclass(frozen=True)
class SimulationResult:
    requests: tuple[RequestResult, ...]
    usage: tuple[UsageInterval, ...]
    prices: tuple[PricePoint, ...]
    arrival_horizon_s: float
    elapsed_s: float


@dataclass(frozen=True)
class ActiveRequestSnapshot:
    """Public fields for an admitted request; excludes simulator-only truth."""

    request_id: str
    admitted_at_s: float
    rate_per_token: float
    tier: str | None
    prompt_tokens: int
    max_tokens: int
    segment: str
    order_type: Literal["limit", "market"]

    @property
    def reserved_kv_tokens(self) -> int:
        return self.prompt_tokens + self.max_tokens


@dataclass(frozen=True)
class SimulationSnapshot:
    """Immutable public state emitted at a settled simulator event."""

    time_s: float
    event: str
    pending: tuple[Order, ...]
    active: tuple[ActiveRequestSnapshot, ...]
    free_capacity: FreeCapacity
    busy: bool
    completed_count: int
    timeout_count: int
    infeasible_count: int
    admitted_count: int
    revenue_observed: float
    latest_clearing_price: PricePoint | None
