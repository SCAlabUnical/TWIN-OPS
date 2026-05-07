# Operational schema rules (\(\mathcal{S}_{OPER}\)) — reference description

This document describes the **descriptor-conditioned operational instantiation rules** used to derive operational digital twin schemas from an admissible schema, consistently with the paper notation.

The simulator implements two generation modes:

- **ADM**: direct sampling from an admissible schema \(\mathcal{S}_{ADM}\)
- **OPER**: single-pass instantiation from an operational schema \(\mathcal{S}_{OPER}\), induced offline

---

## 1. Descriptor \(x\)

Operational specialization is conditioned on a descriptor:

\[
x = (\rho, b),
\]

where:

- \(\rho\) is the load level
- \(b \in \{0,1\}\) is a stress flag (steady vs burst-window workload)

Offline induction and operational instantiation are performed independently for each \(x\).

---

## 2. Seed selection (offline induction, Phase 5)

For each descriptor \(x\), the offline pipeline produces an elite set \(\mathcal{E}_x\).

The operational schema selects a **deterministic representative seed** \(t_{seed}\) from \(\mathcal{E}_x\), using the **medoid** rule under a configuration-space distance (as in the paper).

Ties are broken deterministically (lexicographic order on schema parameters).

---

## 3. Budget-preserving paired perturbations (Phase 6)

Given a seed \(t_{seed}\), the operational schema generates a family of \(K\) variants by applying **paired moves** designed to preserve:

- total server budget \(N_1+N_2\)
- total buffer budget \(B_1+B_2\)

whenever feasible, while remaining within admissible bounds.

If a move violates admissible bounds, a deterministic **repair/clipping** rule is applied.  
If repair is not feasible, the move becomes a deterministic **null-move**.

---

## 4. Uniqueness of the operational family

Because the configuration space is discrete, deterministic paired moves can yield duplicates.

In the paper, an operational family of size \(K\) is interpreted strictly as \(K\) **distinct** twins.  
Uniqueness is enforced over the schema-parameter signature.

For the two-stage pipeline, the signature is:

\[
(N_1, N_2, B_1, B_2).
\]

Variants are generated sequentially and accepted only if their signature is new.

If a duplicate is generated, a deterministic **paired-jitter** fallback is applied to explore a small admissible neighborhood while preserving total budgets. A bounded attempt budget is enforced.

---

## 5. What is guaranteed

For every descriptor \(x\), the operational schema guarantees:

- all generated variants are **admissible** (contract-compatible)
- generation is **single-pass** and deterministic (given the same offline-induced rules)
- operational variants provide controlled diversity around a descriptor-specific seed
