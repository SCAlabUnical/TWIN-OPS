"""
Phase 3: Simulation-based evaluation.

For each candidate twin and descriptor x, run R independent replications
and compute metrics over the measurement window [T_w, T].

Outputs:
- eval_replica.csv  (one row per replication)
- eval_summary.csv  (mean/std over replications)

All time quantities are expressed in milliseconds (ms).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .common import ProgressPrinter, cost_model, list_descriptors, load_yaml, read_jsonl
from .simulator import simulate_replication


def evaluate(
    cfg: Dict[str, Any],
    twins_jsonl_path: str,
    out_replica_csv: str,
    out_summary_csv: str,
    progress: Optional[ProgressPrinter] = None,
) -> None:
    """
    Evaluate the provided twins for all descriptors and write replica- and summary-level CSVs.

    This signature matches the workflow runner (`run_all.py`).
    """
    twins = read_jsonl(Path(twins_jsonl_path))
    descs = list_descriptors(cfg)

    sim_cfg = cfg["sim"]
    R = int(sim_cfg["replications_R"])

    total = len(twins) * len(descs) * R
    if progress is None:
        progress = ProgressPrinter(total=total, label="Phase3-EVAL")
    else:
        # Reinitialize label/total for this phase while keeping global print-rate constraint.
        progress.reset(total=total, label="Phase3-EVAL")

    out_replica_path = Path(out_replica_csv)
    out_summary_path = Path(out_summary_csv)
    out_replica_path.parent.mkdir(parents=True, exist_ok=True)
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    done = 0

    for t in twins:
        # The JSONL rows already include twin_id and parameters.
        for d in descs:
            for r in range(R):
                # IMPORTANT (determinism): the simulator uses the passed `seed` as the
                # `replication_index` inside its own deterministic SHA-256 seeding
                # scheme (see simulator.py). Therefore the replication identifier must
                # be tied to the semantic replication counter `r` (0..R-1), NOT to the
                # running task index. Using a running index makes results depend on the
                # evaluation order (which can change across runs due to filesystem order,
                # different candidate sets, or parallelization), and it cascades into
                # non-deterministic Pareto sets and seeds.
                seed = r
                result = simulate_replication(cfg, t, d.rho, d.burst, seed=seed)
                row = {
                    "domain": cfg["domain"],
                    "source": t.get("source", "unknown"),
                    "twin_id": t["twin_id"],
                    "N1": int(t["N1"]),
                    "N2": int(t["N2"]),
                    "B1": int(t["B1"]),
                    "B2": int(t["B2"]),
                    "cost": float(cost_model(cfg, t)),
                    "rho": float(d.rho),
                    # Descriptor identifiers (keep both legacy and paper-style names)
                    "burst": int(d.burst),
                    "burst_flag": int(d.burst),
                    "desc_id": d.desc_id,
                    "descriptor_id": d.desc_id,
                    "replica": int(r),
                    # Metrics
                    "rt_mean": float(result["rt_mean"]),
                    "rt_p50": float(result.get("rt_p50", 0.0)),
                    "rt_p95": float(result.get("rt_p95", 0.0)),
                    "rt_p99": float(result["rt_p99"]),
                    "rt_max": float(result.get("rt_max", 0.0)),
                    "drop": float(result["drop"]),
                    # Paper-style metric aliases expected by later phases
                    "rt_p50_ms": float(result.get("rt_p50", 0.0)),
                    "rt_p95_ms": float(result.get("rt_p95", 0.0)),
                    "rt_p99_ms": float(result["rt_p99"]),
                    "rt_max_ms": float(result.get("rt_max", 0.0)),
                    "drop_rate": float(result["drop"]),
                    "completed": int(result["completed"]),
                    "arrivals": int(result["arrivals"]),
                    "drops": int(result["drops"]),
                }
                rows.append(row)
                done += 1
                progress.update(1)

    df = pd.DataFrame(rows)
    df.to_csv(out_replica_path, index=False)

    # Summary: mean/std across replications for each (twin, descriptor).
    grp_cols = [
        "domain",
        "twin_id",
        "N1",
        "N2",
        "B1",
        "B2",
        "source",
        "rho",
        "burst",
        "burst_flag",
        "desc_id",
        "descriptor_id",
        "cost",
    ]
    agg = df.groupby(grp_cols).agg(
        rt_mean_mean=("rt_mean", "mean"),
        rt_mean_std=("rt_mean", "std"),
        rt_p50_mean=("rt_p50", "mean"),
        rt_p50_std=("rt_p50", "std"),
        rt_p95_mean=("rt_p95", "mean"),
        rt_p95_std=("rt_p95", "std"),
        rt_p99_mean=("rt_p99", "mean"),
        rt_p99_std=("rt_p99", "std"),
        rt_max_mean=("rt_max", "mean"),
        rt_max_std=("rt_max", "std"),
        drop_mean=("drop", "mean"),
        drop_std=("drop", "std"),
        # Paper-style aliases
        rt_p50_ms_mean=("rt_p50_ms", "mean"),
        rt_p50_ms_std=("rt_p50_ms", "std"),
        rt_p95_ms_mean=("rt_p95_ms", "mean"),
        rt_p95_ms_std=("rt_p95_ms", "std"),
        rt_p99_ms_mean=("rt_p99_ms", "mean"),
        rt_p99_ms_std=("rt_p99_ms", "std"),
        rt_max_ms_mean=("rt_max_ms", "mean"),
        rt_max_ms_std=("rt_max_ms", "std"),
        drop_rate_mean=("drop_rate", "mean"),
        drop_rate_std=("drop_rate", "std"),
        completed_mean=("completed", "mean"),
        arrivals_mean=("arrivals", "mean"),
        drops_mean=("drops", "mean"),
    ).reset_index()
    agg.to_csv(out_summary_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--twins", required=True)
    ap.add_argument("--out_replica", required=True)
    ap.add_argument("--out_summary", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    evaluate(cfg, args.twins, args.out_replica, args.out_summary, progress=None)


if __name__ == "__main__":
    main()
