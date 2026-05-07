"""
Phase 1: Admissible sampling (ADM).

We sample M admissible twins within the configured bounds.
Sampling is uniform over integer ranges for (N1,N2,B1,B2).
"""

from __future__ import annotations

import argparse
import random
from typing import Any, Dict, List

from .common import ProgressPrinter, Twin, lexicographic_key, load_yaml, stable_hash_int, write_jsonl


def sample_adm(cfg: Dict[str, Any], out_path: str, progress: ProgressPrinter) -> None:
    bounds = cfg["sim"]["bounds"]
    M = int(cfg["offline_budgets"]["M_seeds"])
    base_seed = int(cfg["base_seed"])

    rng = random.Random(base_seed)

    twins: List[Dict[str, Any]] = []
    seen = set()

    for i in range(M * 10):  # safety loop to avoid rare duplicates
        if len(twins) >= M:
            break
        N1 = rng.randint(int(bounds["N1"][0]), int(bounds["N1"][1]))
        N2 = rng.randint(int(bounds["N2"][0]), int(bounds["N2"][1]))
        B1 = rng.randint(int(bounds["B1"][0]), int(bounds["B1"][1]))
        B2 = rng.randint(int(bounds["B2"][0]), int(bounds["B2"][1]))
        key = (N1, N2, B1, B2)
        if key in seen:
            continue
        seen.add(key)
        twin_id = f"adm_{len(twins):05d}"
        t = Twin(twin_id=twin_id, N1=N1, N2=N2, B1=B1, B2=B2, meta={"phase": "ADM"})
        twins.append(t.as_dict())
        if len(twins) % 10 == 0 or len(twins) == M:
            progress.maybe_print(f"[Phase 1] Sampled {len(twins)}/{M} ADM twins")

    # Deterministic ordering
    twins_sorted = sorted(twins, key=lambda d: (d["N1"], d["N2"], d["B1"], d["B2"]))
    write_jsonl(out_path, twins_sorted)
    progress.force_print(f"[Phase 1] Wrote {len(twins_sorted)} ADM twins to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    progress = ProgressPrinter()
    sample_adm(cfg, args.out, progress)


if __name__ == "__main__":
    main()
