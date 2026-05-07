"""
Common utilities for dtwin_edge.

All time quantities are expressed in milliseconds (ms).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Optional

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def stable_hash_int(*parts: Any, modulo: int = 2**31 - 1) -> int:
    """
    Deterministic integer hash for reproducible seeding.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int(h.hexdigest(), 16) % modulo


@dataclass(frozen=True)
class Twin:
    """
    Two-stage pipeline twin parameters.

    N1, N2: number of servers at stage 1 and stage 2
    B1, B2: waiting-buffer capacities (FIFO) at stage 1 and stage 2
    """
    twin_id: str
    N1: int
    N2: int
    B1: int
    B2: int
    meta: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "twin_id": self.twin_id,
            "N1": self.N1,
            "N2": self.N2,
            "B1": self.B1,
            "B2": self.B2,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Twin":
        return Twin(
            twin_id=str(d["twin_id"]),
            N1=int(d["N1"]),
            N2=int(d["N2"]),
            B1=int(d["B1"]),
            B2=int(d["B2"]),
            meta=dict(d.get("meta", {})),
        )


def write_jsonl(path: str, items: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


class ProgressPrinter:
    """
    Rate-limited progress printing.

    Hard requirement:
      - progress + ETA messages must be printed to stdout at most every 5 minutes.

    To enforce this *across the whole workflow* (including multiple scripts/modules),
    the printer can persist its last-print timestamp to a shared file specified by
    environment variable `DTWIN_PROGRESS_FILE`.
    """

    def __init__(self, total: int = 0, label: str = "", min_interval_s: int = 300) -> None:
        env_min = os.environ.get("DTWIN_PROGRESS_MIN_S")
        self.min_interval_s = int(env_min) if env_min is not None else int(min_interval_s)
        self._last_print = self._load_last_print()
        self.reset(total=total, label=label)

    def reset(self, total: int, label: str = "") -> None:
        """Reset counters for a new phase."""
        self.total = int(total)
        self.label = str(label)
        self.done = 0
        self.t0 = time.time()

    def update(self, n: int = 1) -> None:
        """Advance progress and maybe print a rate-limited status line."""
        self.done += int(n)
        if self.total <= 0:
            self.maybe_print(f"[{self.label}] done={self.done}")
            return

        elapsed = max(1e-9, time.time() - self.t0)
        rate = self.done / elapsed
        remaining = max(0, self.total - self.done)
        eta_s = remaining / rate if rate > 0 else float("inf")
        pct = 100.0 * self.done / self.total
        self.maybe_print(f"[{self.label}] {self.done}/{self.total} ({pct:.1f}%) | ETA {eta_s/60.0:.1f} min")

    def _load_last_print(self) -> float:
        path = os.environ.get("DTWIN_PROGRESS_FILE")
        if not path:
            return 0.0
        try:
            with open(path, "r", encoding="utf-8") as f:
                return float(f.read().strip() or 0.0)
        except Exception:
            return 0.0

    def _save_last_print(self, ts: float) -> None:
        path = os.environ.get("DTWIN_PROGRESS_FILE")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(ts))
        except Exception:
            # Best-effort only; do not fail the workflow.
            pass

    def maybe_print(self, msg: str) -> None:
        now = time.time()
        if now - self._last_print >= self.min_interval_s:
            print(msg, flush=True)
            self._last_print = now
            self._save_last_print(now)

    def force_print(self, msg: str) -> None:
        """Force a print only if it does not violate the rate limit."""
        self.maybe_print(msg)
def lexicographic_key(t: Twin) -> Tuple[int, int, int, int]:
    return (t.N1, t.N2, t.B1, t.B2)


# ---------------------------------------------------------------------------
# Descriptor utilities (workload scenarios)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Descriptor:
    """
    Workload descriptor x used throughout the paper.

    - rho: target utilization of the reference system (dimensionless)
    - burst: 0/1 flag (0 = no burst, 1 = bursty arrivals)
    - desc_id: stable identifier used in CSV outputs
    """
    desc_id: str
    rho: float
    burst: int


def list_descriptors(cfg: Dict[str, Any]) -> List[Descriptor]:
    """
    Build the list of workload descriptors from the experiment configuration.

    Expected YAML layout:
      sim:
        workload:
          rho_grid: [ ... ]
          burst_flags: [0, 1]
    """
    workload = cfg["sim"]["workload"]
    rhos = list(workload.get("rho_grid", []))
    bursts = list(workload.get("burst_flags", []))
    descs: List[Descriptor] = []
    for rho in rhos:
        for b in bursts:
            desc_id = f"rho={float(rho):.4g}|burst={int(b)}"
            descs.append(Descriptor(desc_id=desc_id, rho=float(rho), burst=int(b)))
    return descs


def cost_model(cfg: Dict[str, Any], t: Dict[str, Any]) -> float:
    """
    Cost model from the appendix: linear cost of capacity and buffer.

    cost(t) = wN * (N1 + N2) + wB * (B1 + B2)
    """
    w = cfg.get("selection", {}).get("cost_weights", {})
    wN = float(w.get("wN", 1.0))
    wB = float(w.get("wB", 0.0))
    return wN * (int(t["N1"]) + int(t["N2"])) + wB * (int(t["B1"]) + int(t["B2"]))
