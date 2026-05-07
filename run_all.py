#!/usr/bin/env python
"""
run_all.py

Executes the full six-phase workflow for a given domain and experiment tier.

Example:
  python run_all.py --domain dc --tier sanity
"""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

from src.common import ProgressPrinter, ensure_dir, load_yaml
from src.phase1_adm import sample_adm
from src.phase2_texp import explore
from src.phase3_simulate import evaluate
from src.phase4_rank import rank_select
from src.phase5_rules import induce
from src.phase6_oper import instantiate
from src.phase7_report import generate_tables


def _select_config(domain: str, tier: str) -> str:
    if domain not in {"dc", "edge"}:
        raise ValueError("domain must be one of: dc, edge")
    if tier not in {"paper", "sanity"}:
        raise ValueError("tier must be one of: paper, sanity")

    name = f"{domain}_{tier}.yaml"
    return str(Path("configs") / "experiments" / name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["dc", "edge"])
    ap.add_argument("--tier", required=True, choices=["paper", "sanity"])
    args = ap.parse_args()

    cfg_path = _select_config(args.domain, args.tier)
    cfg = load_yaml(cfg_path)

    # Create run directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.domain}_{args.tier}_{ts}"
    run_dir = Path("runs") / run_id
    ensure_dir(str(run_dir))

    # Shared progress state across the whole workflow (enforces 5-minute max print rate)
    progress_file = str(run_dir / ".progress_last_print.txt")
    os.environ["DTWIN_PROGRESS_FILE"] = progress_file
    os.environ["DTWIN_PROGRESS_MIN_S"] = "300"
    os.environ["DTWIN_RUN_ID"] = run_id
    progress = ProgressPrinter(min_interval_s=300)
    progress.maybe_print(f"[run_all] Starting run in {run_dir}")

    # Copy config into the run directory
    shutil.copy(cfg_path, run_dir / Path(cfg_path).name)

    # Phase 1
    t_adm = run_dir / "t_adm.jsonl"
    sample_adm(cfg, str(t_adm), progress)

    # Phase 2
    t_exp = run_dir / "t_exp.jsonl"
    explore(cfg, str(t_adm), str(t_exp), progress)

    # Phase 3 (offline evaluation of TEXP)
    eval_replica = run_dir / "eval_replica.csv"
    eval_summary = run_dir / "eval_summary.csv"
    evaluate(cfg, str(t_exp), str(eval_replica), str(eval_summary), progress)

    # Phase 4
    ranked = run_dir / "ranked.csv"
    elite = run_dir / "elite.csv"
    rank_select(cfg, str(eval_summary), str(ranked), str(elite), progress)

    # Phase 5
    rules = run_dir / "rules.json"
    induce(cfg, str(elite), str(rules), progress)

    # Phase 6
    t_oper = run_dir / "t_oper.jsonl"
    instantiate(cfg, str(rules), str(t_oper), progress)

    # Phase 3 again (evaluate OPER)
    eval_oper_dir = run_dir / "eval_oper"
    ensure_dir(str(eval_oper_dir))
    eval_oper_replica = eval_oper_dir / "eval_replica.csv"
    eval_oper_summary = eval_oper_dir / "eval_summary.csv"
    evaluate(cfg, str(t_oper), str(eval_oper_replica), str(eval_oper_summary), progress)

    # Phase 7 (paper tables)
    generate_tables(str(run_dir), cfg_path, progress)

    progress.maybe_print(f"[run_all] Done. Results in {run_dir}")


if __name__ == "__main__":
    main()
