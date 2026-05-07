"""
Phase 3: Simulation-based evaluation (wrapper).

This module exists to keep backward compatibility with the workflow runner (`run_all.py`),
while delegating the actual implementation to `src.phase3_eval`.

The simulator itself is implemented in `src.simulator` and is reused both:
  - during offline induction (Phases 1→5), and
  - after Phase 6 to evaluate OPER.

All time quantities are expressed in milliseconds (ms).
"""

from __future__ import annotations

from .phase3_eval import evaluate, main  # re-export

__all__ = ["evaluate", "main"]


if __name__ == "__main__":
    main()
