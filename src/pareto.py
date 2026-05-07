"""
Pareto filtering utilities (minimization objectives).

We consider point-wise comparisons on named metrics.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def dominates(a: Dict[str, float], b: Dict[str, float], keys: Sequence[str]) -> bool:
    """
    True if a Pareto-dominates b for minimization on all keys.
    """
    better_or_equal = True
    strictly_better = False
    for k in keys:
        if a[k] > b[k]:
            better_or_equal = False
            break
        if a[k] < b[k]:
            strictly_better = True
    return better_or_equal and strictly_better


def pareto_front(rows: List[Dict[str, float]], keys: Sequence[str]) -> List[bool]:
    """
    Return a boolean mask indicating which rows are on the Pareto front.
    Complexity O(n^2) is fine for the paper's budgets (<= 500 candidates/descriptor).
    """
    n = len(rows)
    is_front = [True] * n
    for i in range(n):
        if not is_front[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if not is_front[i]:
                break
            if dominates(rows[j], rows[i], keys):
                is_front[i] = False
    return is_front
