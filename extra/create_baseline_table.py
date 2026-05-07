#!/usr/bin/env python3
"""

Assumptions:
  - Inputs are always in the local working directory:
      DC:
        .\\dc\\eval_replica.csv
        .\\dc\\eval_oper\\eval_replica.csv
      EDGE:
        .\\edge\\eval_replica.csv
        .\\edge\\eval_oper\\eval_replica.csv
  - K = 20
  - prints ONE aggregated row per condition, per domain

CLI parameters:
  --tol   Budget matching tolerance (default 0.05 -> 5%)
  --seed  RNG seed (kept for reproducibility of any future extensions; currently not used)

Outputs:
  - prints a paper-ready aggregated table (one row per domain & condition)
  - writes a CSV named: baseline_table.csv

Baselines produced:
  - ADM    : exactly as in the simulator logs (no simulator changes required)
  - OPER   : K operational variants per descriptor (as encoded in OPER twin_id)
  - ADM-bestK@budget : selects K ADM twins under a descriptor-wise cost budget matched to OPER,
                       using Pareto-first selection on (RT_p99, DROP) and deterministic tie-breaking

"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


# ----------------------------
# Metrics extraction / naming
# ----------------------------
def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of {candidates} found in columns: {list(df.columns)}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy with canonical column names:
      - rt_p99_ms
      - drop_rate
      - cost
      - descriptor_id
      - twin_id
      - replica
    """
    out = df.copy()

    out["rt_p99_ms"] = out[_pick_col(out, ["rt_p99_ms", "rt_p99", "rt99", "rt_p99 (ms)"])]
    out["drop_rate"] = out[_pick_col(out, ["drop_rate", "drop", "DROP", "drop (rate)"])]
    out["cost"] = out[_pick_col(out, ["cost", "Cost", "COST"])]

    for must in ["descriptor_id", "twin_id", "replica"]:
        if must not in out.columns:
            raise ValueError(f"Required column '{must}' not found")

    return out


def oper_family_filter(oper_df: pd.DataFrame, descriptor_id: str) -> pd.Series:
    """
    Identify the K OPER variants that belong to this descriptor's operational family.

    Expected OPER twin_id format (as in your runs):
      'oper_<descriptor_id>_<suffix>'
    """
    prefix = f"oper_{descriptor_id}_"
    return oper_df["twin_id"].astype(str).str.startswith(prefix)


# ----------------------------
# Aggregation: replica -> twin -> condition
# ----------------------------
@dataclass(frozen=True)
class AggRow:
    mean_rt_p99: float
    inter_sd_rt_p99: float
    mean_intra_sd_rt_p99: float
    mean_drop: float
    inter_sd_drop: float
    mean_intra_sd_drop: float
    sum_cost: float
    mean_cost: float
    n_twins: int


def aggregate_for_twins(df: pd.DataFrame, twin_ids: List[str]) -> AggRow:
    """
    Aggregate metrics for a specific list of twin_ids within df.
    df must already be normalized (rt_p99_ms, drop_rate, cost, descriptor_id, twin_id, replica).
    """
    sub = df[df["twin_id"].isin(twin_ids)].copy()
    if sub.empty:
        raise ValueError("No rows found for the selected twin_ids")

    # cost is constant per twin across replicas; take first
    per_twin_cost = sub.groupby("twin_id")["cost"].first()

    # point estimate per twin: mean over replicas
    per_twin_mean = sub.groupby("twin_id").agg(
        rt_p99_ms=("rt_p99_ms", "mean"),
        drop_rate=("drop_rate", "mean"),
    )

    # intra-twin sd: std over replicas for each twin
    per_twin_intra = sub.groupby("twin_id").agg(
        intra_rt=("rt_p99_ms", "std"),
        intra_drop=("drop_rate", "std"),
    ).fillna(0.0)

    # inter-twin sd: std across the selected twins of their point estimates
    inter_rt = float(per_twin_mean["rt_p99_ms"].std(ddof=0)) if len(per_twin_mean) > 0 else 0.0
    inter_drop = float(per_twin_mean["drop_rate"].std(ddof=0)) if len(per_twin_mean) > 0 else 0.0

    return AggRow(
        mean_rt_p99=float(per_twin_mean["rt_p99_ms"].mean()),
        inter_sd_rt_p99=inter_rt,
        mean_intra_sd_rt_p99=float(per_twin_intra["intra_rt"].mean()),
        mean_drop=float(per_twin_mean["drop_rate"].mean()),
        inter_sd_drop=inter_drop,
        mean_intra_sd_drop=float(per_twin_intra["intra_drop"].mean()),
        sum_cost=float(per_twin_cost.sum()),
        mean_cost=float(per_twin_cost.mean()),
        n_twins=int(per_twin_mean.shape[0]),
    )


# ----------------------------
# Pareto ranking (two objectives, both minimized)
# ----------------------------
def pareto_rank(points: np.ndarray) -> np.ndarray:
    """
    Non-dominated sorting for 2D minimization.
    Returns an array rank[i] where 0 = first Pareto front.
    """
    n = points.shape[0]
    dominates = [set() for _ in range(n)]
    dominated_by_count = np.zeros(n, dtype=int)

    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            i_dom_j = (pi[0] <= pj[0] and pi[1] <= pj[1]) and (pi[0] < pj[0] or pi[1] < pj[1])
            j_dom_i = (pj[0] <= pi[0] and pj[1] <= pi[1]) and (pj[0] < pi[0] or pj[1] < pi[1])
            if i_dom_j:
                dominates[i].add(j)
                dominated_by_count[j] += 1
            elif j_dom_i:
                dominates[j].add(i)
                dominated_by_count[i] += 1

    rank = np.full(n, -1, dtype=int)
    current = [i for i in range(n) if dominated_by_count[i] == 0]
    r = 0
    while current:
        for i in current:
            rank[i] = r
            for j in dominates[i]:
                dominated_by_count[j] -= 1
        current = [j for j in range(n) if dominated_by_count[j] == 0 and rank[j] == -1]
        r += 1

    rank[rank == -1] = r
    return rank


# ----------------------------
# Selection: ADM-bestK@budget(OPER) (Pareto-first, deterministic, budget-matched)
# ----------------------------
def select_bestk_budget(
    adm_twin_table: pd.DataFrame,
    K: int,
    target_budget: float,
    tol: float,
    pool_size: int = 300,
) -> List[str]:
    """
    Select K twins from adm_twin_table subject to approximate budget equality:
        |sum_cost - target_budget| <= tol * target_budget

    adm_twin_table: one row per twin with columns:
        twin_id, cost, rt_p99_ms, drop_rate

    Selection philosophy:
      - Pareto-first: lower Pareto rank on (rt_p99_ms, drop_rate)
      - deterministic tie-breaks (cost, twin_id)
      - then greedy swaps to meet the budget constraint (while trying to keep Pareto quality)
    """
    if adm_twin_table.shape[0] < K:
        raise ValueError(f"Not enough ADM twins to select K={K} (have {adm_twin_table.shape[0]})")

    pts = adm_twin_table[["rt_p99_ms", "drop_rate"]].to_numpy(dtype=float)
    ranks = pareto_rank(pts)

    t = adm_twin_table.copy()
    t["pareto_rank"] = ranks
    t = t.sort_values(["pareto_rank", "cost", "twin_id"], ascending=[True, True, True])

    pool = t.head(min(pool_size, len(t))).copy()
    if pool.shape[0] < K:
        pool = t.copy()

    selected = pool.head(K).copy()

    def budget_err(sel: pd.DataFrame) -> float:
        return float(sel["cost"].sum() - target_budget)

    max_abs_err = tol * target_budget
    err = budget_err(selected)
    if abs(err) <= max_abs_err:
        return selected["twin_id"].tolist()

    out = pool[~pool["twin_id"].isin(set(selected["twin_id"].tolist()))].copy()
    selected = selected.reset_index(drop=True)
    out = out.reset_index(drop=True)

    for _ in range(2000):
        err = budget_err(selected)
        if abs(err) <= max_abs_err:
            break

        need_decrease = err > 0  # selected too expensive
        if need_decrease:
            rem_idx = selected.sort_values(["pareto_rank", "cost"], ascending=[False, False]).index[0]
            rem = selected.loc[rem_idx]
            cand = out[out["cost"] < rem["cost"]]
        else:
            rem_idx = selected.sort_values(["pareto_rank", "cost"], ascending=[False, True]).index[0]
            rem = selected.loc[rem_idx]
            cand = out[out["cost"] > rem["cost"]]

        if cand.empty:
            if pool.shape[0] < t.shape[0]:
                pool = t.copy()
                out = pool[~pool["twin_id"].isin(set(selected["twin_id"].tolist()))].reset_index(drop=True)
                continue
            break

        worst_rank = int(selected["pareto_rank"].max())
        cand2 = cand[cand["pareto_rank"] <= worst_rank + 1]
        if cand2.empty:
            cand2 = cand  # relax

        current_sum = float(selected["cost"].sum())
        new_sums = current_sum - float(rem["cost"]) + cand2["cost"].astype(float)
        cand2 = cand2.assign(_abs_budget_err=(new_sums - target_budget).abs())
        cand2 = cand2.sort_values(["_abs_budget_err", "pareto_rank", "cost", "twin_id"],
                                  ascending=[True, True, True, True]).head(1)
        add = cand2.iloc[0]

        selected = selected.drop(index=rem_idx).reset_index(drop=True)
        out = out[out["twin_id"] != add["twin_id"]].reset_index(drop=True)
        selected = pd.concat([selected, add.to_frame().T], ignore_index=True)
        out = pd.concat([out, rem.to_frame().T], ignore_index=True)

    final_sum = float(selected["cost"].sum())
    final_err = final_sum - target_budget
    if abs(final_err) > max_abs_err:
        print(
            f"[WARN] Could not match budget within tolerance. "
            f"target={target_budget:.6f}, got={final_sum:.6f}, "
            f"abs_err={abs(final_err):.6f}, allowed={max_abs_err:.6f}",
            file=sys.stderr,
        )

    return selected["twin_id"].tolist()


# ----------------------------
# Per-domain computation (per descriptor) -> then paper aggregation
# ----------------------------
def compute_domain_rows(adm_raw: pd.DataFrame, oper_raw: pd.DataFrame, domain: str, K: int, tol: float) -> pd.DataFrame:
    adm = normalize_columns(adm_raw)
    oper = normalize_columns(oper_raw)

    descriptors = sorted(set(adm["descriptor_id"].unique()) & set(oper["descriptor_id"].unique()))
    out_rows = []

    for did in descriptors:
        adm_sub = adm[adm["descriptor_id"] == did]
        oper_sub = oper[oper["descriptor_id"] == did]

        # OPER family twins for this descriptor
        mask = oper_family_filter(oper_sub, did)
        oper_family_twins = sorted(oper_sub[mask]["twin_id"].unique().tolist())

        # if mismatch, fall back deterministically by cost
        if len(oper_family_twins) < K:
            oper_costs = oper_sub.groupby("twin_id")["cost"].first().sort_values()
            oper_family_twins = oper_costs.index.tolist()[:K]
        elif len(oper_family_twins) > K:
            oper_costs = oper_sub[oper_sub["twin_id"].isin(oper_family_twins)].groupby("twin_id")["cost"].first().sort_values()
            oper_family_twins = oper_costs.index.tolist()[:K]

        # Target budget = sum cost of those K OPER twins
        B = float(oper_sub[oper_sub["twin_id"].isin(oper_family_twins)].groupby("twin_id")["cost"].first().sum())

        # Build per-twin table for ADM (means over replicas)
        adm_tw = (
            adm_sub.groupby("twin_id")
            .agg(cost=("cost", "first"),
                 rt_p99_ms=("rt_p99_ms", "mean"),
                 drop_rate=("drop_rate", "mean"))
            .reset_index()
            .sort_values(["twin_id"])
        )

        best_ids = select_bestk_budget(adm_tw, K=K, target_budget=B, tol=tol, pool_size=300)

        # Aggregate rows (paper semantics)
        row_oper = aggregate_for_twins(oper_sub, oper_family_twins)
        row_best = aggregate_for_twins(adm_sub, best_ids)

        # Full ADM row (kept, since your paper already reports ADM and you don't want simulator changes)
        adm_all_twins = sorted(adm_sub["twin_id"].unique().tolist())
        row_adm = aggregate_for_twins(adm_sub, adm_all_twins)

        def add(condition: str, r: AggRow):
            out_rows.append({
                "domain": domain,
                "descriptor_id": did,
                "condition": condition,
                "K": r.n_twins,
                "mean_rt_p99_ms": r.mean_rt_p99,
                "inter_sd_rt_p99_ms": r.inter_sd_rt_p99,
                "mean_intra_sd_rt_p99_ms": r.mean_intra_sd_rt_p99,
                "mean_drop": r.mean_drop,
                "inter_sd_drop": r.inter_sd_drop,
                "mean_intra_sd_drop": r.mean_intra_sd_drop,
                "sum_cost": r.sum_cost,
                "mean_cost": r.mean_cost,
                "target_budget_oper": B,
                "abs_budget_err": abs(r.sum_cost - B),
                "rel_budget_err": abs(r.sum_cost - B) / B if B > 0 else 0.0,
                "tol": tol,
            })

        add("ADM", row_adm)
        add("OPER", row_oper)
        add(f"ADM-bestK@budget(tol={tol:.2f})", row_best)

    return pd.DataFrame(out_rows)


def aggregate_table_like_paper(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-descriptor rows into a single row per (domain, condition).
    Arithmetic mean across descriptor_id for each metric (each descriptor weighted equally).
    """
    metric_cols = [
        "mean_rt_p99_ms", "inter_sd_rt_p99_ms", "mean_intra_sd_rt_p99_ms",
        "mean_drop", "inter_sd_drop", "mean_intra_sd_drop",
        "mean_cost", "sum_cost", "target_budget_oper", "abs_budget_err", "rel_budget_err"
    ]
    keep = [c for c in metric_cols if c in df.columns]

    g = (df.groupby(["domain", "condition"], as_index=False)[keep]
           .mean(numeric_only=True))

    mx = (df.groupby(["domain", "condition"], as_index=False)["rel_budget_err"]
            .max()
            .rename(columns={"rel_budget_err": "rel_budget_err_max"}))
    g = g.merge(mx, on=["domain", "condition"], how="left")
    g = g.rename(columns={"rel_budget_err": "rel_budget_err_mean"})
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.05, help="Budget matching tolerance (e.g., 0.05 = 5%)")
    ap.add_argument("--seed", type=int, default=0, help="Seed (kept for reproducibility; currently unused)")
    args = ap.parse_args()

    # Fixed assumptions
    K = 20

    dc_adm_path = os.path.join(".", "dc", "eval_replica.csv")
    dc_oper_path = os.path.join(".", "dc", "eval_oper", "eval_replica.csv")
    edge_adm_path = os.path.join(".", "edge", "eval_replica.csv")
    edge_oper_path = os.path.join(".", "edge", "eval_oper", "eval_replica.csv")

    for p in [dc_adm_path, dc_oper_path, edge_adm_path, edge_oper_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required input file: {p}")

    dc_adm = pd.read_csv(dc_adm_path)
    dc_oper = pd.read_csv(dc_oper_path)
    edge_adm = pd.read_csv(edge_adm_path)
    edge_oper = pd.read_csv(edge_oper_path)

    df_dc = compute_domain_rows(dc_adm, dc_oper, domain="dc", K=K, tol=args.tol)
    df_edge = compute_domain_rows(edge_adm, edge_oper, domain="edge", K=K, tol=args.tol)

    df = pd.concat([df_dc, df_edge], ignore_index=True)
    agg = aggregate_table_like_paper(df)

    # Print paper-style aggregated rows
    cols = ["domain", "condition",
            "mean_rt_p99_ms", "inter_sd_rt_p99_ms", "mean_intra_sd_rt_p99_ms",
            "mean_drop", "inter_sd_drop", "mean_intra_sd_drop",
            "mean_cost", "rel_budget_err_mean", "rel_budget_err_max"]
    pd.set_option("display.max_rows", 200)
    print(agg[cols].to_string(index=False))

    # Write outputs
    out_csv = "baseline_table.csv"
    agg.to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()
