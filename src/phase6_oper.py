"""
Phase 6: Single-pass instantiation of operational families (OPER).

Given the deterministic rules induced in Phase 5, we generate K operational twins
around the per-descriptor seed using deterministic "paired moves" that preserve totals:

- Move 1 server from upstream to downstream (N1-1, N2+1)
- Move 1 server from downstream to upstream (N1+1, N2-1)
- Move 5 buffer units from upstream to downstream (B1-5, B2+5)
- Move 5 buffer units from downstream to upstream (B1+5, B2-5)

Moves are attempted in a fixed cyclic order until K variants are produced.
Deterministic repair is applied if a move would violate bounds:
- Clip the violating component to its bound
- Apply compensation to preserve the total within the budget group (N_tot or B_tot)
- If compensation would violate bounds too, the move becomes a null-move

Output:
- t_oper.jsonl: one record per generated twin, with metadata linking to descriptor and seed.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Tuple

from .common import ProgressPrinter, Twin, load_yaml, write_jsonl


def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _paired_move(seed: Dict[str, int], move: str, bounds: Dict[str, List[int]], totals: Dict[str, int]) -> Dict[str, int]:
    """
    Apply a paired move with deterministic repair, preserving totals per budget group.
    """
    N1, N2, B1, B2 = seed["N1"], seed["N2"], seed["B1"], seed["B2"]
    Ntot = totals["N_tot"]
    Btot = totals["B_tot"]

    N1_lo, N1_hi = int(bounds["N1"][0]), int(bounds["N1"][1])
    N2_lo, N2_hi = int(bounds["N2"][0]), int(bounds["N2"][1])
    B1_lo, B1_hi = int(bounds["B1"][0]), int(bounds["B1"][1])
    B2_lo, B2_hi = int(bounds["B2"][0]), int(bounds["B2"][1])

    dN = 1
    dB = 5

    if move == "N_upstream_to_downstream":
        N1_new = N1 - dN
        N2_new = N2 + dN
        # Repair with clipping/compensation
        N1_new_c = _clip(N1_new, N1_lo, N1_hi)
        N2_new_c = Ntot - N1_new_c
        if not (N2_lo <= N2_new_c <= N2_hi):
            return {"N1": N1, "N2": N2, "B1": B1, "B2": B2}
        return {"N1": N1_new_c, "N2": N2_new_c, "B1": B1, "B2": B2}

    if move == "N_downstream_to_upstream":
        N1_new = N1 + dN
        N1_new_c = _clip(N1_new, N1_lo, N1_hi)
        N2_new_c = Ntot - N1_new_c
        if not (N2_lo <= N2_new_c <= N2_hi):
            return {"N1": N1, "N2": N2, "B1": B1, "B2": B2}
        return {"N1": N1_new_c, "N2": N2_new_c, "B1": B1, "B2": B2}

    if move == "B_upstream_to_downstream":
        B1_new = B1 - dB
        B1_new_c = _clip(B1_new, B1_lo, B1_hi)
        B2_new_c = Btot - B1_new_c
        if not (B2_lo <= B2_new_c <= B2_hi):
            return {"N1": N1, "N2": N2, "B1": B1, "B2": B2}
        return {"N1": N1, "N2": N2, "B1": B1_new_c, "B2": B2_new_c}

    if move == "B_downstream_to_upstream":
        B1_new = B1 + dB
        B1_new_c = _clip(B1_new, B1_lo, B1_hi)
        B2_new_c = Btot - B1_new_c
        if not (B2_lo <= B2_new_c <= B2_hi):
            return {"N1": N1, "N2": N2, "B1": B1, "B2": B2}
        return {"N1": N1, "N2": N2, "B1": B1_new_c, "B2": B2_new_c}

    return {"N1": N1, "N2": N2, "B1": B1, "B2": B2}



def _signature(t: Dict[str, int], domain: str = "dc") -> Tuple[int, int, int, int, int]:
    """Return a hashable signature for uniqueness checks."""
    n1, n2, b1, b2 = int(t["N1"]), int(t["N2"]), int(t["B1"]), int(t["B2"])
    # Network buffer is fixed in configs for EDGE; keep field for future-proofing.
    bnet = int(t.get("Bnet", -1))
    return (n1, n2, b1, b2, bnet if domain == "edge" else -1)


def _jitter_candidate(
    base: Dict[str, int],
    bounds: Dict[str, List[int]],
    totals: Dict[str, int],
    delta_n_choices: List[int],
    delta_b_choices: List[int],
) -> List[Dict[str, int]]:
    """Generate a deterministic sequence of jittered candidates around a base solution.

    Jitter preserves totals by paired changes:
    - N1 += dN, N2 -= dN
    - B1 += dB, B2 -= dB

    Candidates are clipped/compensated to respect bounds. If a jitter would violate
    bounds after compensation, it is skipped.
    """
    out: List[Dict[str, int]] = []
    Ntot = int(totals["N_tot"])
    Btot = int(totals["B_tot"])
    N1_lo, N1_hi = int(bounds["N1"][0]), int(bounds["N1"][1])
    N2_lo, N2_hi = int(bounds["N2"][0]), int(bounds["N2"][1])
    B1_lo, B1_hi = int(bounds["B1"][0]), int(bounds["B1"][1])
    B2_lo, B2_hi = int(bounds["B2"][0]), int(bounds["B2"][1])

    base_n1, base_n2, base_b1, base_b2 = int(base["N1"]), int(base["N2"]), int(base["B1"]), int(base["B2"])

    # Deterministic ordering: loop delta_n outer, delta_b inner.
    for dN in delta_n_choices:
        n1 = _clip(base_n1 + int(dN), N1_lo, N1_hi)
        n2 = Ntot - n1
        if not (N2_lo <= n2 <= N2_hi):
            continue
        for dB in delta_b_choices:
            b1 = _clip(base_b1 + int(dB), B1_lo, B1_hi)
            b2 = Btot - b1
            if not (B2_lo <= b2 <= B2_hi):
                continue
            out.append({"N1": n1, "N2": n2, "B1": b1, "B2": b2})
    return out



def instantiate(cfg: Dict[str, Any], rules_json: str, out_oper_jsonl: str, progress: ProgressPrinter) -> None:
    """Instantiate operational families (OPER) from induced rules.

    By default, this function preserves the legacy behaviour (deterministic walk with paired moves).
    If cfg.oper.enforce_unique is True, it enforces K *distinct* twins per descriptor (uniqueness
    over the discrete decision variables) using a deterministic jitter mechanism (Option C1).
    """
    K = int(cfg["offline_budgets"]["K_oper_family"])
    bounds = cfg["sim"]["bounds"]
    move_order = list(cfg["paired_moves_order"])
    domain = str(cfg.get("domain", "dc"))

    oper_cfg = cfg.get("oper", {}) if isinstance(cfg.get("oper", {}), dict) else {}
    enforce_unique = bool(oper_cfg.get("enforce_unique", False))
    max_attempts = int(oper_cfg.get("max_attempts", 2000 if enforce_unique else K))
    # Deterministic jitter (Option C1). Preserve totals by paired changes.
    delta_n_choices = list(oper_cfg.get("jitter_N", [-1, 1]))
    delta_b_choices = list(oper_cfg.get("jitter_B", [-2, -1, 1, 2]))
    jitter_tries = int(oper_cfg.get("jitter_tries", 20))

    with open(rules_json, "r", encoding="utf-8") as f:
        rules = json.load(f)

    out: List[Dict[str, Any]] = []

    for desc_id in sorted(rules.keys()):
        seed = rules[desc_id]["seed"]
        totals = rules[desc_id]["totals"]

        base = {"N1": int(seed["N1"]), "N2": int(seed["N2"]), "B1": int(seed["B1"]), "B2": int(seed["B2"])}
        seed_twin_id = str(seed.get("twin_id", "unknown"))

        selected: List[Dict[str, int]] = []
        seen = set()

        # Always start from the seed itself.
        selected.append(base)
        seen.add(_signature(base, domain))

        current = base
        attempts = 0
        move_idx = 0

        while len(selected) < K and attempts < max_attempts:
            attempts += 1
            move = move_order[move_idx % len(move_order)]
            move_idx += 1

            cand = _paired_move(current, move, bounds, totals)
            move_used = move

            sig = _signature(cand, domain)
            if (not enforce_unique) or (sig not in seen):
                selected.append(cand)
                seen.add(sig)
                current = cand
                continue

            # Duplicate: try deterministic jitter around the candidate (Option C1).
            # We generate a small deterministic neighbourhood and pick the first unseen candidate.
            neighbourhood = _jitter_candidate(
                base=cand,
                bounds=bounds,
                totals=totals,
                delta_n_choices=delta_n_choices,
                delta_b_choices=delta_b_choices,
            )

            found = False
            for k, jc in enumerate(neighbourhood):
                if k >= jitter_tries:
                    break
                jsig = _signature(jc, domain)
                if jsig in seen:
                    continue
                selected.append(jc)
                seen.add(jsig)
                current = jc
                move_used = f"{move}+jitter"
                found = True
                break

            if not found:
                # Could not find a new point around this candidate; continue walking.
                # Keeping the state unchanged is safer and deterministic.
                current = cand

        # Emit OPER twins
        for j, variant in enumerate(selected):
            tid = f"oper_{desc_id}_{j:03d}"
            out.append(Twin(
                twin_id=tid,
                N1=int(variant["N1"]), N2=int(variant["N2"]),
                B1=int(variant["B1"]), B2=int(variant["B2"]),
                meta={
                    "phase": "OPER",
                    "descriptor_id": desc_id,
                    "seed_twin_id": seed_twin_id,
                    "variant_index": j,
                    "unique_enforced": enforce_unique,
                },
            ).as_dict())

        if enforce_unique and len(selected) < K:
            progress.force_print(
                f"[Phase 6][WARN] {desc_id}: only {len(selected)}/{K} unique OPER twins generated "
                f"after max_attempts={max_attempts}. Consider increasing bounds or max_attempts."
            )
        else:
            progress.maybe_print(f"[Phase 6] {desc_id}: generated K={len(selected)} OPER twins")

    write_jsonl(out_oper_jsonl, out)
    progress.force_print(f"[Phase 6] Wrote {len(out)} OPER twins to {out_oper_jsonl}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    progress = ProgressPrinter()
    instantiate(cfg, args.rules, args.out, progress)


if __name__ == "__main__":
    main()
