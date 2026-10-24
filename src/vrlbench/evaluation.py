"""Compatibility imports for callers of the earlier evaluation API.

Execution lives only in :mod:`vrlbench.simulation`; no parallel evaluator or
trial loop is maintained here.
"""

from .simulation import plan, simulate as evaluate

__all__ = ["evaluate", "plan"]
