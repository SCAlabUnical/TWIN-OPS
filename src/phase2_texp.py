"""
Phase 2: Bounded admissible exploration (TEXP).

For each sampled ADM twin, generate up to V admissible variants following the deterministic
enumeration rule:

  (N1 ± dN), (N2 ± dN), (B1 ± dB), (B2 ± dB)

where dN=1, dB=5 by default.

Invalid variants (out of bounds) and duplicates are discarded.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Set, Tuple

from .common import ProgressPrinter, Twin, load_yaml, read_jsonl, write_jsonl


def _in_bounds(v: int, lo: int, hi: int) -> bool:
    return lo <= v <= hi


def explore(cfg: Dict[str, Any], adm_path: str, out_path: str, progress: ProgressPrinter) -> None:
    bounds = cfg["sim"]["bounds"]
    V = int(cfg["offline_budgets"]["V_variants_per_seed"])
    dN = int(cfg["exploration_steps"]["dN"])
    dB = int(cfg["exploration_steps"]["dB"])

    adm_dicts = read_jsonl(adm_path)
    adm_twins = [Twin.from_dict(d) for d in adm_dicts]

    candidates: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, int, int, int]] = set()

    def add_twin(N1: int, N2: int, B1: int, B2: int, meta: Dict[str, Any]) -> None:
        key = (N1, N2, B1, B2)
        if key in seen:
            return
        seen.add(key)
        tid = f"t_{len(candidates):06d}"
        candidates.append(Twin(twin_id=tid, N1=N1, N2=N2, B1=B1, B2=B2, meta=meta).as_dict())

    # Include ADM seeds in TEXP
    for t in adm_twins:
        add_twin(t.N1, t.N2, t.B1, t.B2, {"phase": "ADM", "seed_id": t.twin_id})

    for idx, seed in enumerate(adm_twins):
        produced = 0
        # Fixed deterministic enumeration order
        moves = [
            ("N1", +dN), ("N1", -dN),
            ("N2", +dN), ("N2", -dN),
            ("B1", +dB), ("B1", -dB),
            ("B2", +dB), ("B2", -dB),
        ]
        for (field, delta) in moves:
            if produced >= V:
                break
            N1, N2, B1, B2 = seed.N1, seed.N2, seed.B1, seed.B2
            if field == "N1":
                N1 += delta
            elif field == "N2":
                N2 += delta
            elif field == "B1":
                B1 += delta
            elif field == "B2":
                B2 += delta

            if not (_in_bounds(N1, *bounds["N1"]) and _in_bounds(N2, *bounds["N2"]) and
                    _in_bounds(B1, *bounds["B1"]) and _in_bounds(B2, *bounds["B2"])):
                continue

            add_twin(N1, N2, B1, B2, {"phase": "TEXP", "seed_id": seed.twin_id, "move": f"{field}{delta:+d}"})
            produced += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == len(adm_twins):
            progress.maybe_print(f"[Phase 2] Processed {idx+1}/{len(adm_twins)} seeds; "
                                 f"total candidates={len(candidates)}")

    # Deterministic ordering
    candidates_sorted = sorted(candidates, key=lambda d: (d["N1"], d["N2"], d["B1"], d["B2"], d["twin_id"]))
    write_jsonl(out_path, candidates_sorted)
    progress.force_print(f"[Phase 2] Wrote {len(candidates_sorted)} candidates to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adm", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    progress = ProgressPrinter()
    explore(cfg, args.adm, args.out, progress)


if __name__ == "__main__":
    main()
