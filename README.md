Reproducible **Python + SimPy** implementation of the **six-phase methodology** described in the paper:
L. Belcastro, F. Marozzo, D. Talia, P. Trunfio, "Operational Digital Twin Instantiation for Edge-to-Cloud Service Platforms", submitted to IEEE Smart World Congress 2026.

The workflow supports two domains:

- **Data Center** (`dc`)
- **Edge** (`edge`) — adds a fixed network delay and a finite network buffer between Stage 1 and Stage 2.

The pipeline performs:

- Phase 1: sample initial twins (ADM)
- Phase 2: explore / generate candidate twins (TEXP)
- Phase 3: offline simulation-based evaluation of TEXP (all descriptors)
- Phase 4: rank candidates and select elite per descriptor
- Phase 5: induce operational rules (deterministic)
- Phase 6: instantiate operational families (OPER)
- Phase 3 again: evaluate OPER (final evaluation)
- Phase 7: generate paper tables from Phase 3 outputs (results + operability)


Progress is printed to stdout, rate-limited to at most once every 5 minutes.

---

## 1) Install

Create and activate a virtual environment (recommended), then install the required libraries:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 2) Experiment tiers

Two tiers per domain:

### A) Paper-grade (parameters as in the paper; can take hours)
- `dtwin_eval/configs/experiments/dc_paper.yaml`
- `dtwin_eval/configs/experiments/edge_paper.yaml`

### B) Sanity-check (small numbers; finishes in minutes)
- `dtwin_eval/configs/experiments/dc_sanity.yaml`
- `dtwin_eval/configs/experiments/edge_sanity.yaml`

---

## 3) Run the full pipeline

Use the provided **Python-only** runner script:

After each run, Phase 7 automatically produces the paper tables under `paper_tables/`.
You can also re-generate tables for an existing run without re-running simulations:

```bash
python -m src.phase7_report --run-dir runs/<timestamped_run_dir> --config configs/experiments/<domain>_<tier>.yaml
```

### Sanity runs (fast)

**Data Center sanity**
```bash
python run_all.py --domain dc --tier sanity
```

**Edge sanity**
```bash
python run_all.py --domain edge --tier sanity
```

### Paper-grade runs (long)

**Data Center paper**
```bash
python run_all.py --domain dc --tier paper
```

**Edge paper**
```bash
python run_all.py --domain edge --tier paper
```

Each run creates a timestamped folder under `runs/` and executes the full workflow.

---


### OPER uniqueness

An operational family has **size K made of K distinct twins**
(with respect to the discrete decision variables). To avoid degenerate families and artificially low
inter-family variability, **Phase 6 enforces uniqueness** by default using a deterministic *jitter* mechanism:

- If a newly generated OPER twin duplicates a previously selected one, we apply small paired perturbations
  (N1+=d, N2-=d; B1+=d, B2-=d) within bounds, trying a fixed neighbourhood until a unique configuration is found.

You can control this behaviour in the experiment YAML files under `oper`:

```yaml
oper:
  enforce_unique: true
  max_attempts: 2000     # paper (use smaller values for sanity)
  jitter_tries: 20
  jitter_N: [-1, 1]
  jitter_B: [-2, -1, 1, 2]
```

If you want to disable uniqueness enforcement, set `oper.enforce_unique: false`.

## 4) Outputs

Inside each run directory:

### Offline induction artifacts
- `t_adm.jsonl` (Phase 1)
- `t_exp.jsonl` (Phase 2)
- `eval_replica.csv`, `eval_summary.csv` (Phase 3)
- `ranked.csv`, `elite.csv` (Phase 4)
- `rules.json` (Phase 5)

### Operational families
- `t_oper.jsonl` (Phase 6)

### Final evaluation (post-Phase6)
- `eval_oper/eval_replica.csv`, `eval_oper/eval_summary.csv`

### Paper tables (Phase 7)
- `paper_tables/dc_results.csv` or `paper_tables/edge_results.csv`
- `paper_tables/dc_operability.csv` or `paper_tables/edge_operability.csv`

---

## Notes on reproducibility

- Every simulation replication uses deterministic seeds derived from:
  - a `base_seed` from the YAML configuration
  - the descriptor id
  - the twin id
  - the replication index
- Tie-breaking rules in Phase 4 and Phase 5 are deterministic (lexicographic on `(N1,N2,B1,B2)`).

---

## Folder layout

- `run_all.py` executes the whole workflow
- `configs/` contains YAML configuration files
- `src/` contains Python programs implementing workflow phases
- `runs/` contains timestamped subfolders, one of each run executed
- `extra/` contains scripts to generate goodput figures and to calculate baseline table

## Paper parameter alignment

The `*_paper.yaml` experiment tiers are aligned to the numerical constants reported in:

- **Case Study I (Data Center)**: Table I
  - Service means: m1=5 ms, m2=10 ms
  - Bounds: N_k in [1,12], B_k in [0,50]
  - Reference capacity: N2_ref=6
  - Horizon / warm-up: T=50,000 ms, Tw=5,000 ms
  - Burst schedule: P=5,000 ms, W=1,000 ms, burst factor F=3
  - Load regimes: rho in {0.75, 0.90}
  - Replications: R=30
  - Operational family size: K=20
- **Case Study II (Edge)**: Table III
  - Service means: m1=8 ms, m2=12 ms
  - Network transfer: d_net=5 ms, B_net=20
  - Bounds: N_k in [1,6], B_k in [0,30]
  - Reference capacity: N2_ref=3
  - Other workload and protocol constants: as in Case Study I

Offline induction constants:
- Cost weights: wN=0.8, wB=0.2
- Elite size cap: E=10
- Offline budgets: M=50 seeds, V=10 variants per seed (<=500 evaluated candidates)

Contract semantics and metric definitions:
- deterministic same-timestamp event ordering (completions -> arrivals -> admissions),
- FIFO dispatching with deterministic tie-breaking,
- stage-1 admission/drop only,
- blocking-after-service transfers (and edge network in-flight stage),
- RT_p99 (nearest-rank) and DROP computed over [Tw, T].

### Additional notes
- burst windows start with the HIGH-rate window at t=0 for duration W, then LOW-rate for P-W, repeating every P;
- the Phase-6 paired-move order is controlled by the YAML key `paired_moves_order`.


## Reproducibility package

This repository provides:

- a reference simulator implementation
- configuration-driven experiments (ADM vs OPER)
- raw per-replication outputs
- scripts to regenerate paper tables/figures

Documentation:

- Contract semantics: `contract.md`
- Operational schema rules: `operational_schema.md`

## Extra folder

The extra folder contains two subfolders:

- dc\, under which the following files must be inserted (copied from a reference dc run): eval_replica.csv and \eval_oper\eval_replica.cvs
- edge\, under which the following files must be inserted (copied from a reference edge run): eval_replica.csv and \eval_oper\eval_replica.cvs

and two python scripts:

- create_goodput_figures.py: creates dc_goodput.png and edge_goodput.png
- create_baseline_table.py: creates baseline_table.csv

The scripts can be executed with the following commands:

```bash
python create_figures.py
```

```bash
python create_baseline_table.py --tol 0.05 --seed 0
```
