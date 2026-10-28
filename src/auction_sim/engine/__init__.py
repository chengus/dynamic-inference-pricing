"""Continuous-batching event engine and public entry points."""

from .simulator import PolicyError, Simulator, simulate

__all__ = ["PolicyError", "Simulator", "simulate"]
