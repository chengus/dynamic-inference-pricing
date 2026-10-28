"""Synthetic, reproducible request streams for the auction simulator."""

from .generator import DemandConfig, Workload, make_workload

__all__ = ["DemandConfig", "Workload", "make_workload"]
