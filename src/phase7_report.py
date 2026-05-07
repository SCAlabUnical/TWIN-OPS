r"""
Phase 7: Paper tables generation.

This phase only aggregates the CSV artifacts produced by Phase 3
(offline induction evaluation) and the post-Phase6 Phase 3 evaluation (OPER evaluation) to build
the paper tables.

Input artifacts (inside a run directory):
  - eval_replica.csv
  - eval_summary.csv
  - eval_oper/eval_replica.csv
  - eval_oper/eval_summary.csv

Output artifacts (written under run_dir/paper_tables):
  - dc_results.csv / edge_results.csv
  - dc_operability.csv / edge_operability.csv

All time units are milliseconds (ms).

The aggregation follows the paper definitions:

Let g be a scalar indicator (RT_p99 or DROP). For each twin t and descriptor x:

  \bar{g}(t;x) is the mean across R replications  (available as *_mean columns in eval_summary.csv)
  \sigma^{intra}_g(t;x) is the std across R replications (available as *_std columns in eval_summary.csv)

For a family of K twins (ADM or OPER), and for a fixed descriptor x:

  \overline{\sigma^{intra}_g}(x) = mean_t \sigma^{intra}_g(t;x)
  \sigma^{inter}_g(x) = std_t \bar{g}(t;x)

The tables in the paper show a single number per condition. We therefore report the average across
all descriptors x in X:

  \overline{\sigma^{intra}_g} = mean_x \overline{\sigma^{intra}_g}(x)
  \sigma^{inter}_g = mean_x \sigma^{inter}_g(x)

For the point estimates, we report:
  - point estimate = mean_{x,t} \bar{g}(t;x)
  - uncertainty     = mean_{x,t} \sigma^{intra}_g(t;x)

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd

from .common import ProgressPrinter, ensure_dir, load_yaml, read_jsonl


def _load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "twin_id", "descriptor_id", "rho", "burst_flag",
        "rt_p99_ms_mean", "rt_p99_ms_std", "drop_rate_mean", "drop_rate_std",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df



def _load_twin_ids_by_phase(jsonl_path: str) -> Dict[str, set]:
    """Load twin_ids grouped by meta.phase from a twins JSONL file."""
    groups: Dict[str, set] = {}
    if not Path(jsonl_path).exists():
        return groups
    for obj in read_jsonl(jsonl_path):
        phase = str(obj.get("meta", {}).get("phase", "UNKNOWN"))
        tid = str(obj.get("twin_id"))
        if not tid:
            continue
        groups.setdefault(phase, set()).add(tid)
    return groups



def _results_table(df_sum: pd.DataFrame) -> Tuple[float, float, float, float]:
    """Return (rt_mean, rt_std, drop_mean, drop_std) for a condition.

    Paper-style aggregation:
      - For each descriptor x: compute mean_t and std_t across the K twins of the per-twin means (already
        averaged across R replications and stored as *_mean columns).
      - Then average these per-descriptor aggregates across x in X.
    """
    per_x = df_sum.groupby(["descriptor_id", "rho", "burst_flag"], as_index=False).agg(
        rt_mean=("rt_p99_ms_mean", "mean"),
        rt_std=("rt_p99_ms_mean", "std"),
        drop_mean=("drop_rate_mean", "mean"),
        drop_std=("drop_rate_mean", "std"),
    )
    # If K=1 for some x, std is NaN; treat as 0 (no variability across twins).
    per_x = per_x.fillna({"rt_std": 0.0, "drop_std": 0.0})

    rt_mean = float(per_x["rt_mean"].mean())
    rt_std = float(per_x["rt_std"].mean())
    drop_mean = float(per_x["drop_mean"].mean())
    drop_std = float(per_x["drop_std"].mean())
    return rt_mean, rt_std, drop_mean, drop_std


def _operability_table(df_sum: pd.DataFrame) -> Tuple[float, float, float, float]:
    """Return (sigma_intra_rt, sigma_intra_d, sigma_inter_rt, sigma_inter_d) averaged across descriptors."""
    # For each descriptor x, compute intra mean over twins and inter std over twins.
    per_x = df_sum.groupby(["descriptor_id", "rho", "burst_flag"], as_index=False).agg(
        sigma_intra_rt=("rt_p99_ms_std", "mean"),
        sigma_intra_d=("drop_rate_std", "mean"),
        sigma_inter_rt=("rt_p99_ms_mean", "std"),
        sigma_inter_d=("drop_rate_mean", "std"),
    )

    # If K=1 for some x, std is NaN; treat as 0 (no inter variability).
    per_x = per_x.fillna({"sigma_inter_rt": 0.0, "sigma_inter_d": 0.0})

    sigma_intra_rt = float(per_x["sigma_intra_rt"].mean())
    sigma_intra_d = float(per_x["sigma_intra_d"].mean())
    sigma_inter_rt = float(per_x["sigma_inter_rt"].mean())
    sigma_inter_d = float(per_x["sigma_inter_d"].mean())
    return sigma_intra_rt, sigma_intra_d, sigma_inter_rt, sigma_inter_d




def _secondary_metrics_table(df_sum: pd.DataFrame, meas_window_ms: float) -> Dict[str, float]:
    """Compute secondary metrics for a condition, averaged across descriptors (paper-style).

    Metrics reported (all averaged across descriptors in X):
      - RT_mean_ms: mean across K twins (each twin averaged across R replications) then averaged across descriptors
      - RT_p50_ms, RT_p95_ms, RT_max_ms: same aggregation as RT_mean_ms
      - THROUGHPUT_rps: completed requests per second in measurement window
      - GOODPUT_rps: completed requests per second, multiplied by acceptance (1 - DROP)
      - ACCEPT_RATE: 1 - DROP
      - COMPLETION_RATIO: completed / arrivals in window (can be < accept rate if horizon ends before completion)
    """
    if meas_window_ms <= 0:
        meas_window_ms = 1.0

    # Point-style aggregates for response-time moments/percentiles.
    per_x_rt = df_sum.groupby(["descriptor_id", "rho", "burst_flag"], as_index=False).agg(
        rt_mean=("rt_mean_mean", "mean"),
        rt_p50=("rt_p50_ms_mean", "mean") if "rt_p50_ms_mean" in df_sum.columns else ("rt_mean_mean", "mean"),
        rt_p95=("rt_p95_ms_mean", "mean") if "rt_p95_ms_mean" in df_sum.columns else ("rt_p99_ms_mean", "mean"),
        rt_max=("rt_max_ms_mean", "mean") if "rt_max_ms_mean" in df_sum.columns else ("rt_p99_ms_mean", "mean"),
        drop=("drop_rate_mean", "mean"),
        completed=("completed_mean", "mean"),
        arrivals=("arrivals_mean", "mean") if "arrivals_mean" in df_sum.columns else ("completed_mean", "mean"),
    )

    # Convert completed/ms -> rps
    throughput_rps = (per_x_rt["completed"] / meas_window_ms) * 1000.0
    accept_rate = 1.0 - per_x_rt["drop"]
    goodput_rps = throughput_rps * accept_rate

    completion_ratio = per_x_rt["completed"] / per_x_rt["arrivals"].replace({0.0: float('nan')})
    completion_ratio = completion_ratio.fillna(0.0)

    return {
        "RT_mean_ms": float(per_x_rt["rt_mean"].mean()),
        "RT_p50_ms": float(per_x_rt["rt_p50"].mean()),
        "RT_p95_ms": float(per_x_rt["rt_p95"].mean()),
        "RT_max_ms": float(per_x_rt["rt_max"].mean()),
        "THROUGHPUT_rps": float(throughput_rps.mean()),
        "GOODPUT_rps": float(goodput_rps.mean()),
        "ACCEPT_RATE": float(accept_rate.mean()),
        "COMPLETION_RATIO": float(completion_ratio.mean()),
    }

def generate_tables(run_dir: str, config_path: str, progress: ProgressPrinter) -> str:
    """Generate a single Phase-7 CSV table for the paper.

    Output: <run_dir>/paper_tables/table.csv with columns:
      Condition, RT_p99, sigma_inter_RT, sigma_intra_RT, DROP, sigma_inter_D, sigma_intra_D

    The ADM row aggregates ONLY the initial Phase-1 seed twins (phase tag 'ADM' from t_exp.jsonl).
    OPER aggregates the post-Phase-6 evaluation in eval_oper/eval_summary.csv.
    """
    runp = Path(run_dir)
    cfg = load_yaml(config_path)
    out_dir = runp / "paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    domain = str(cfg.get("domain", "dc")).lower()
    if domain not in {"dc", "edge"}:
        raise ValueError(f"Unknown domain in config: {domain}")

    # Offline induction evaluation (contains both Phase-1 seeds and explored variants).
    offline_sum = _load_summary(str(runp / "eval_summary.csv"))
    phases = _load_twin_ids_by_phase(str(runp / "t_exp.jsonl"))
    adm_ids = phases.get("ADM", set())
    if len(adm_ids) == 0:
        # Best-effort fallback (older runs): treat the whole offline set as ADM.
        adm_sum = offline_sum
    else:
        adm_sum = offline_sum[offline_sum["twin_id"].isin(sorted(adm_ids))].copy()

    # OPER (post-Phase-6 evaluation)
    oper_sum = _load_summary(str(runp / "eval_oper" / "eval_summary.csv"))

    # Point estimates (mean across K twins of per-twin means) + inter-twin std across K
    adm_rt_m, adm_rt_s, adm_d_m, adm_d_s = _results_table(adm_sum)
    oper_rt_m, oper_rt_s, oper_d_m, oper_d_s = _results_table(oper_sum)

    # Operability summaries: intra = avg over twins of within-twin std; inter = std across twins of per-twin means
    adm_intra_rt, adm_intra_d, adm_inter_rt, adm_inter_d = _operability_table(adm_sum)
    oper_intra_rt, oper_intra_d, oper_inter_rt, oper_inter_d = _operability_table(oper_sum)

    df_table = pd.DataFrame([
        {
            "Condition": "ADM",
            "RT_p99": adm_rt_m,
            "sigma_inter_RT": adm_inter_rt,
            "sigma_intra_RT": adm_intra_rt,
            "DROP": adm_d_m,
            "sigma_inter_D": adm_inter_d,
            "sigma_intra_D": adm_intra_d,
        },
        {
            "Condition": "OPER",
            "RT_p99": oper_rt_m,
            "sigma_inter_RT": oper_inter_rt,
            "sigma_intra_RT": oper_intra_rt,
            "DROP": oper_d_m,
            "sigma_inter_D": oper_inter_d,
            "sigma_intra_D": oper_intra_d,
        },
    ])

    table_path = out_dir / "table.csv"
    df_table.to_csv(table_path, index=False)

    # Secondary metrics table (optional, does not affect paper table.csv)
    meas_window_ms = float(cfg.get("sim", {}).get("horizon_ms", 0.0)) - float(cfg.get("sim", {}).get("warmup_ms", 0.0))
    adm_sec = _secondary_metrics_table(adm_sum, meas_window_ms)
    oper_sec = _secondary_metrics_table(oper_sum, meas_window_ms)
    df_secondary = pd.DataFrame([
        {"Condition": "ADM", **adm_sec},
        {"Condition": "OPER", **oper_sec},
    ])
    sec_path = out_dir / "secondary_metrics.csv"
    df_secondary.to_csv(sec_path, index=False)

    progress.force_print(f"[Phase 7] Wrote {table_path}")
    return str(out_dir)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Run directory produced by run_all.py (timestamped folder).")
    ap.add_argument("--config", required=True, help="Experiment YAML configuration used for the run.")
    args = ap.parse_args()

    progress = ProgressPrinter()
    generate_tables(args.run_dir, args.config, progress)


if __name__ == "__main__":
    main()
