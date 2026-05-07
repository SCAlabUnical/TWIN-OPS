"""
Phase 4: Multi-criteria selection with cost-aware size control.

For each descriptor x:
1) Compute Pareto non-dominated set over primitive indicators.
2) If |P_x| > E, keep the E lowest-cost candidates (deterministic tie-breaking).

Primitive indicators (minimize all):
- rt_p99_mean
- drop_mean
- cost

Outputs:
- ranked.csv (all candidates with dominance + elite flags)
- elite.csv  (selected elite per descriptor)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .common import ensure_dir, lex_key, load_yaml, pareto_nondominated


def select_elite(cfg: Dict[str, Any], summary_csv: Path, outdir: Path) -> None:
    E = int(cfg["offline_budgets"]["E"])
    df = pd.read_csv(summary_csv)

    ranked_rows: List[Dict[str, Any]] = []
    elite_rows: List[Dict[str, Any]] = []

    # Process each descriptor separately.
    for (rho, burst), g in df.groupby(["rho", "burst"], sort=True):
        rows = g.to_dict(orient="records")
        nd_flags = pareto_nondominated(rows, objective_fields=["rt_p99_mean", "drop_mean", "cost"])

        for r, nd in zip(rows, nd_flags):
            r["nondominated"] = bool(nd)

        pareto = [r for r in rows if r["nondominated"]]
        # Cost-aware size control.
        pareto.sort(key=lambda r: (float(r["cost"]), int(r["N1"]), int(r["N2"]), int(r["B1"]), int(r["B2"])))

        elite = pareto if len(pareto) <= E else pareto[:E]
        elite_keys = {(int(r["N1"]), int(r["N2"]), int(r["B1"]), int(r["B2"])) for r in elite}

        for r in rows:
            r["elite"] = (int(r["N1"]), int(r["N2"]), int(r["B1"]), int(r["B2"])) in elite_keys
            ranked_rows.append(r)
            if r["elite"]:
                elite_rows.append(r)

    ranked_df = pd.DataFrame(ranked_rows)
    ranked_df.to_csv(outdir / "ranked.csv", index=False)

    elite_df = pd.DataFrame(elite_rows)
    elite_df.to_csv(outdir / "elite.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to experiment YAML.")
    ap.add_argument("--outdir", required=True, help="Run output directory.")
    ap.add_argument("--summary", required=True, help="Path to eval_summary.csv.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    outdir = ensure_dir(args.outdir)
    select_elite(cfg, Path(args.summary), outdir)


if __name__ == "__main__":
    main()
