"""Synthetic diurnal arrivals, optional interactive-agent bursts, and private demand.

The generator makes no hardware-performance claims. Independent random streams
ensure changing output lengths or values never changes arrival times.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from ..models import Job, Order


@dataclass(frozen=True)
class DemandConfig:
    rate_rps: float = 0.5
    interactive_fraction: float = 0.5
    diurnal_amplitude: float = 0.25
    diurnal_period_s: float = 86400.0
    prompt_lognormal_mean: float = 4.7
    prompt_lognormal_sigma: float = 0.55
    output_lognormal_mean: float = 5.3
    output_lognormal_sigma: float = 0.55
    agent_prompt_lognormal_mean: float | None = 4.5
    agent_prompt_lognormal_sigma: float | None = 0.55
    agent_output_lognormal_mean: float | None = 4.8
    agent_output_lognormal_sigma: float | None = 0.55
    output_cap_tokens: int = 512
    reservation_slack: float = 1.25
    interactive_value_lognormal_mean: float = -3.0
    agent_value_lognormal_mean: float = -4.0
    value_lognormal_sigma: float = 0.65
    agent_rate_rps: float = 0.0
    agent_active_to_idle_s: float = 0.20
    agent_idle_to_active_s: float = 0.025
    agent_rate_multiplier: float = 5.0
    agent_max_wait_s: float = 300.0
    market_max_wait_s: float = 0.0
    max_prompt_tokens: int = 2048

    def __post_init__(self) -> None:
        nonnegative = ("rate_rps", "agent_rate_rps", "prompt_lognormal_sigma",
                       "output_lognormal_sigma", "value_lognormal_sigma", "market_max_wait_s",
                       "agent_max_wait_s")
        for key in nonnegative:
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        for key in ("prompt_lognormal_mean", "output_lognormal_mean", "interactive_value_lognormal_mean",
                    "agent_value_lognormal_mean", "agent_active_to_idle_s", "agent_idle_to_active_s",
                    "agent_rate_multiplier"):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite")
        for key in ("agent_prompt_lognormal_mean", "agent_output_lognormal_mean"):
            value = getattr(self, key)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if not math.isfinite(self.diurnal_amplitude) or not 0 <= self.diurnal_amplitude < 1:
            raise ValueError("diurnal_amplitude must be in [0, 1)")
        if not math.isfinite(self.interactive_fraction) or not 0 <= self.interactive_fraction <= 1:
            raise ValueError("interactive_fraction must be in [0, 1]")
        if self.diurnal_period_s <= 0 or not math.isfinite(self.diurnal_period_s):
            raise ValueError("diurnal_period_s must be finite and positive")
        for key in ("agent_prompt_lognormal_sigma", "agent_output_lognormal_sigma"):
            value = getattr(self, key)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{key} must be finite and nonnegative")
        for key in ("output_cap_tokens", "max_prompt_tokens"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("token caps must be positive integers")
        if not math.isfinite(self.reservation_slack) or self.reservation_slack < 1:
            raise ValueError("reservation_slack must be finite and at least 1")
        if self.agent_active_to_idle_s <= 0 or self.agent_idle_to_active_s <= 0:
            raise ValueError("MMPP transition rates must be positive")
        if self.agent_rate_multiplier < 1:
            raise ValueError("agent_rate_multiplier must be at least 1")


@dataclass(frozen=True)
class Workload:
    jobs: tuple[Job, ...]
    seed: int
    horizon_s: float
    arrivals: tuple[float, ...]
    metadata: dict[str, Any]


def _nhpp_arrivals(rng: np.random.Generator, rate: float, amplitude: float,
                   period: float, horizon: float) -> list[float]:
    if horizon <= 0 or rate <= 0:
        return []
    bound = rate * (1 + amplitude)
    n = int(rng.poisson(bound * horizon))
    candidates = np.sort(rng.uniform(0.0, horizon, size=n))
    return [float(t) for t in candidates if rng.random() <=
            (1 + amplitude * math.sin(2 * math.pi * t / period)) / (1 + amplitude)]


def _mmpp_arrivals(rng: np.random.Generator, cfg: DemandConfig, horizon: float) -> list[float]:
    """Two-state MMPP overlay with configurable transition hazards."""
    if horizon <= 0 or cfg.agent_rate_rps <= 0:
        return []
    # Stationary initialization makes mean-preserving burst sweeps valid even
    # for finite horizons, without an artificial initial idle period.
    active_probability = cfg.agent_idle_to_active_s / (cfg.agent_idle_to_active_s + cfg.agent_active_to_idle_s)
    t, active, out = 0.0, bool(rng.random() < active_probability), []
    while t < horizon:
        transition = cfg.agent_idle_to_active_s if not active else cfg.agent_active_to_idle_s
        arrival_rate = cfg.agent_rate_rps * (cfg.agent_rate_multiplier if active else 1.0)
        wait = rng.exponential(1.0 / (transition + arrival_rate))
        t += float(wait)
        if t >= horizon:
            break
        if rng.random() < transition / (transition + arrival_rate):
            active = not active
        else:
            out.append(t)
    return out


def make_workload(config: DemandConfig, seed: int, horizon_s: float) -> Workload:
    """Generate a paired-policy workload; bids and caps are public truthfully."""
    if not math.isfinite(horizon_s) or horizon_s < 0:
        raise ValueError("horizon_s must be finite and nonnegative")
    seed_seq = np.random.SeedSequence(seed)
    interactive_rng, agent_base_rng, mmpp_rng, length_rng, value_rng = [
        np.random.default_rng(s) for s in seed_seq.spawn(5)]
    interactive_rate = config.rate_rps * config.interactive_fraction
    agent_base_rate = config.rate_rps * (1 - config.interactive_fraction)
    interactive = _nhpp_arrivals(interactive_rng, interactive_rate, config.diurnal_amplitude,
                                 config.diurnal_period_s, horizon_s)
    agent_base = _nhpp_arrivals(agent_base_rng, agent_base_rate, config.diurnal_amplitude,
                                config.diurnal_period_s, horizon_s)
    agent_bursts = _mmpp_arrivals(mmpp_rng, config, horizon_s)
    arrivals = sorted([(t, "interactive") for t in interactive] +
                      [(t, "agent") for t in agent_base + agent_bursts])
    jobs: list[Job] = []
    for index, (arrival, segment) in enumerate(arrivals):
        agent = segment == "agent"
        prompt_mean = config.agent_prompt_lognormal_mean if agent and config.agent_prompt_lognormal_mean is not None else config.prompt_lognormal_mean
        prompt_sigma = config.agent_prompt_lognormal_sigma if agent and config.agent_prompt_lognormal_sigma is not None else config.prompt_lognormal_sigma
        output_mean = config.agent_output_lognormal_mean if agent and config.agent_output_lognormal_mean is not None else config.output_lognormal_mean
        output_sigma = config.agent_output_lognormal_sigma if agent and config.agent_output_lognormal_sigma is not None else config.output_lognormal_sigma
        prompt = int(np.clip(np.rint(length_rng.lognormal(prompt_mean,
                                                         prompt_sigma)),
                             1, config.max_prompt_tokens))
        output = int(np.clip(np.rint(length_rng.lognormal(output_mean,
                                                         output_sigma)),
                             1, config.output_cap_tokens))
        max_tokens = max(output, int(math.ceil(config.output_cap_tokens * config.reservation_slack)))
        value_mean = config.interactive_value_lognormal_mean if segment == "interactive" else config.agent_value_lognormal_mean
        value = float(value_rng.lognormal(value_mean, config.value_lognormal_sigma))
        is_agent = agent
        order = Order(
            request_id=f"s{seed}-r{index:08d}", arrival_s=arrival,
            prompt_tokens=prompt, max_tokens=max_tokens, bid_per_token=value,
            max_wait_s=config.agent_max_wait_s if is_agent else config.market_max_wait_s,
            segment=segment, order_type="limit" if is_agent else "market",
        )
        jobs.append(Job(order, output, value))
    return Workload(tuple(jobs), int(seed), float(horizon_s),
                    tuple(job.order.arrival_s for job in jobs),
                    {"demand": asdict(config), "n_requests": len(jobs),
                     "synthetic": True, "arrival_model": "split_diurnal_nhpp_plus_agent_two_state_mmpp"})
