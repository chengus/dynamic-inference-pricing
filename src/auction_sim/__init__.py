"""Admission auction simulator. Importing this package never starts a simulation."""

from .engine import PolicyError, Simulator, simulate
from .oracle import OracleResult, OracleSearchLimit, exact_hindsight_oracle

__version__ = "0.1.0"

__all__ = [
    "OracleResult", "OracleSearchLimit", "PolicyError", "Simulator",
    "exact_hindsight_oracle", "simulate",
]
