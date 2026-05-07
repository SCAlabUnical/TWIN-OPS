"""
Phase 5: Offline induction of operational rules (deterministic).

For each descriptor x, we compute a representative seed t_seed from the elite set E_x:
- Medoid under normalized L1 distance on (N1,N2,B1,B2)
- Ties broken lexicographically by (N1,N2,B1,B2)

Output:
- rules.json: mapping descriptor_id -> seed twin parameters + totals for later OPER generation
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Tuple

import pandas as pd

from .common import ProgressPrinter, Twin, lexicographic_key, load_yaml


def _dist_l1_norm(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int],
                  Ntot_range: Tuple[int, int], Btot_range: Tuple[int, int]) -> float:
    N1, N2, B1, B2 = a
    N1p, N2p, B1p, B2p = b
    Nmin_tot, Nmax_tot = Ntot_range
    Bmin_tot, Bmax_tot = Btot_range
    dn = (abs(N1 - N1p) + abs(N2 - N2p)) / (Nmax_tot - Nmin_tot) if (Nmax_tot - Nmin_tot) > 0 else 0.0
    db = (abs(B1 - B1p) + abs(B2 - B2p)) / (Bmax_tot - Bmin_tot) if (Bmax_tot - Bmin_tot) > 0 else 0.0
    return float(dn + db)


def induce(cfg: Dict[str, Any], elite_csv: str, out_rules_json: str, progress: ProgressPrinter) -> None:
    bounds = cfg["sim"]["bounds"]
    Ntot_range = (2 * int(bounds["N1"][0]), 2 * int(bounds["N1"][1]))
    Btot_range = (2 * int(bounds["B1"][0]), 2 * int(bounds["B1"][1]))

    df = pd.read_csv(elite_csv)
    rules: Dict[str, Any] = {}

    for desc_id, gdf in df.groupby("descriptor_id", sort=True):
        twins: List[Twin] = []
        for _, row in gdf.iterrows():
            twins.append(Twin(
                twin_id=str(row["twin_id"]),
                N1=int(row["N1"]), N2=int(row["N2"]), B1=int(row["B1"]), B2=int(row["B2"]),
                meta={"phase": "ELITE", "descriptor_id": desc_id}
            ))

        # Compute medoid: argmin over sum distances
        best = None
        best_sum = None
        for t in twins:
            s = 0.0
            a = (t.N1, t.N2, t.B1, t.B2)
            for u in twins:
                b = (u.N1, u.N2, u.B1, u.B2)
                s += _dist_l1_norm(a, b, Ntot_range, Btot_range)
            if best_sum is None or s < best_sum or (abs(s - best_sum) < 1e-12 and lexicographic_key(t) < lexicographic_key(best)):  # type: ignore[arg-type]
                best = t
                best_sum = s

        assert best is not None
        rules[desc_id] = {
            "seed": {"N1": best.N1, "N2": best.N2, "B1": best.B1, "B2": best.B2, "twin_id": best.twin_id},
            "totals": {"N_tot": best.N1 + best.N2, "B_tot": best.B1 + best.B2},
        }
        progress.maybe_print(f"[Phase 5] {desc_id}: selected seed {best.twin_id} (N1={best.N1},N2={best.N2},B1={best.B1},B2={best.B2})")

    with open(out_rules_json, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, sort_keys=True)
    progress.force_print(f"[Phase 5] Wrote operational rules to {out_rules_json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--elite", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    progress = ProgressPrinter()
    induce(cfg, args.elite, args.out, progress)


if __name__ == "__main__":
    main()
