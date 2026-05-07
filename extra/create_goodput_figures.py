#!/usr/bin/env python3
"""
Generate paper-ready Goodput violin plots (ADM vs OPER) for EDGE and DC.

Outputs:
  - edge_goodput.png
  - dc_goodput.png

Requirements:
  pip install pandas numpy matplotlib
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def compute_goodput(df: pd.DataFrame, meas_seconds: float) -> pd.DataFrame:
    """
    goodput_rps = throughput_rps * (1 - drop)
    throughput_rps = completed / meas_seconds
    """
    required = {"completed", "drop"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in CSV")

    out = df.copy()
    out["throughput_rps"] = out["completed"] / float(meas_seconds)
    out["goodput_rps"] = out["throughput_rps"] * (1.0 - out["drop"])
    return out


def twin_level_by_descriptor(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (descriptor_id, twin_id), averaging across replicas.
    """
    required = {"descriptor_id", "twin_id", "goodput_rps"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} after goodput computation")

    return (
        df.groupby(["descriptor_id", "twin_id"], as_index=False)
          .agg(goodput_rps=("goodput_rps", "mean"))
    )


def violin_goodput_compact_points(
    adm_td: pd.DataFrame,
    oper_td: pd.DataFrame,
    out_png: str,
    rng_seed: int = 0,
    max_points_per_violin: int = 180,
) -> None:
    """
    Paper-style: descriptor labels only; ADM grey, OPER orange; violin + points; no title.
    """
    descs = sorted(set(adm_td["descriptor_id"].unique()) | set(oper_td["descriptor_id"].unique()))

    # --- tighter layout: less whitespace between descriptors + closer ADM/OPER ---
    group_spacing = 0.85   # distance between successive descriptors (was 2.0)
    offset = 0.18         # ADM vs OPER separation within a descriptor (was 0.8)
    violin_width = 0.35   # narrower violins help when groups are closer
    jitter = 0.06         # tighter jitter reduces clutter

    pos: list[float] = []
    data: list[np.ndarray] = []

    for i, did in enumerate(descs):
        a = adm_td.loc[adm_td["descriptor_id"] == did, "goodput_rps"].dropna().to_numpy()
        o = oper_td.loc[oper_td["descriptor_id"] == did, "goodput_rps"].dropna().to_numpy()
        data.extend([a, o])
        center = group_spacing * i
        pos.extend([center - offset, center + offset])

    plt.figure(figsize=(8.2, 5.1))
    parts = plt.violinplot(
        data,
        positions=pos,
        widths=violin_width,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    # Color violins: ADM grey, OPER orange (alternating)
    for j, body in enumerate(parts["bodies"]):
        if j % 2 == 0:
            body.set_facecolor("lightgrey")
            body.set_edgecolor("grey")
        else:
            body.set_facecolor("orange")
            body.set_edgecolor("orange")
        body.set_alpha(0.75)

    # Jittered points (subsampled)
    rng = np.random.default_rng(rng_seed)
    for p, vals in zip(pos, data):
        if len(vals) == 0:
            continue
        vv = vals
        if len(vv) > max_points_per_violin:
            vv = rng.choice(vv, size=max_points_per_violin, replace=False)
        x = p + rng.uniform(-jitter, jitter, size=len(vv))
        plt.scatter(x, vv, s=6, alpha=0.25, color="black")

    # Compact x ticks: descriptor only, centered between ADM and OPER
    centers = [group_spacing * i for i in range(len(descs))]
    plt.xticks(centers, descs)
    plt.xlabel("Workload descriptor")
    plt.ylabel("Goodput (req/s)")
    plt.grid(True, alpha=0.25, axis="y")

    # Legend
    adm_patch = mpatches.Patch(color="lightgrey", label="ADM")
    oper_patch = mpatches.Patch(color="orange", label="OPER")
    plt.legend(handles=[adm_patch, oper_patch], loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=240)
    plt.close()


def build_plot(
    adm_csv: str,
    oper_csv: str,
    meas_seconds: float,
    out_png: str,
) -> None:
    adm = pd.read_csv(adm_csv)
    oper = pd.read_csv(oper_csv)

    adm = compute_goodput(adm, meas_seconds)
    oper = compute_goodput(oper, meas_seconds)

    adm_td = twin_level_by_descriptor(adm)
    oper_td = twin_level_by_descriptor(oper)

    violin_goodput_compact_points(adm_td, oper_td, out_png)


def main() -> None:
    # === EDIT THESE PATHS ===
    EDGE_ADM_CSV = "edge/eval_replica.csv"
    EDGE_OPER_CSV = "edge/eval_oper/eval_replica.csv"
    DC_ADM_CSV   = "dc/eval_replica.csv"
    DC_OPER_CSV  = "dc/eval_oper/eval_replica.csv"

    # Measurement interval (seconds): horizon - warmup, consistent with your runs
    EDGE_MEAS_SECONDS = 1.0  # <-- set correctly
    DC_MEAS_SECONDS   = 1.0  # <-- set correctly

    os.makedirs("figures", exist_ok=True)

    build_plot(
        adm_csv=EDGE_ADM_CSV,
        oper_csv=EDGE_OPER_CSV,
        meas_seconds=EDGE_MEAS_SECONDS,
        out_png="edge_goodput.png",
    )
    build_plot(
        adm_csv=DC_ADM_CSV,
        oper_csv=DC_OPER_CSV,
        meas_seconds=DC_MEAS_SECONDS,
        out_png="dc_goodput.png",
    )

    print("Done. Figures saved in the present directory")


if __name__ == "__main__":
    main()
