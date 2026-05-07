"""
Phase 4: Rank candidates and select elite per descriptor.

Procedure per descriptor x:
1) Pareto-filter candidates using primitive objectives:
   - minimize rt_p99_ms_mean
   - minimize drop_rate_mean
2) If Pareto set size > E, sort by:
   - cost (ascending)
   - then lexicographic (N1,N2,B1,B2)
   and keep the first E as elite.
3) Ranked output contains all Pareto candidates with rank order.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import pandas as pd

from .common import ProgressPrinter, load_yaml
from .pareto import pareto_front


def rank_select(cfg: Dict[str, Any], eval_summary_csv: str, out_ranked_csv: str, out_elite_csv: str,
                progress: ProgressPrinter) -> None:
    E = int(cfg["selection"]["elite_size_E"])
    obj_keys = ["rt_p99_ms_mean", "drop_rate_mean"]

    df = pd.read_csv(eval_summary_csv)
    ranked_rows: List[pd.DataFrame] = []
    elite_rows: List[pd.DataFrame] = []

    for desc_id, gdf in df.groupby("descriptor_id", sort=True):
        records = gdf[obj_keys].to_dict(orient="records")
        mask = pareto_front(records, obj_keys)
        pdf = gdf.loc[mask].copy()

        # Deterministic ranking
        pdf = pdf.sort_values(by=["cost", "N1", "N2", "B1", "B2", "twin_id"], ascending=True)
        pdf["rank"] = range(1, len(pdf) + 1)

        ranked_rows.append(pdf)

        if len(pdf) <= E:
            elite = pdf.copy()
        else:
            elite = pdf.head(E).copy()

        elite_rows.append(elite)
        progress.maybe_print(f"[Phase 4] {desc_id}: pareto={len(pdf)} elite={len(elite)}")

    out_ranked = pd.concat(ranked_rows, axis=0, ignore_index=True) if ranked_rows else df.head(0)
    out_elite = pd.concat(elite_rows, axis=0, ignore_index=True) if elite_rows else df.head(0)

    out_ranked.to_csv(out_ranked_csv, index=False)
    out_elite.to_csv(out_elite_csv, index=False)
    progress.force_print(f"[Phase 4] Wrote {out_ranked_csv} and {out_elite_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-summary", required=True)
    ap.add_argument("--out-ranked", required=True)
    ap.add_argument("--out-elite", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    progress = ProgressPrinter()
    rank_select(cfg, args.eval_summary, args.out_ranked, args.out_elite, progress)


if __name__ == "__main__":
    main()
