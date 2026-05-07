# Contract semantics (Twin Simulation Contract \(\mathcal{C}\))

This document describes the **contract-level execution semantics** used by the simulator.

The goal is to make the simulator behavior **unambiguous and reproducible**, while keeping the contract independent from any specific implementation.

---

## 1. Time and event ordering

Time is continuous.

Events are processed in **non-decreasing timestamp** order.  
For events with the **same timestamp**, the following precedence is applied:

1. **Service completions**
2. **Arrivals**
3. **Admission/drop decisions** (stage 1 only)

---

## 2. Stages, capacity, and FIFO queues

Each service stage \(k\) consists of:

- \(N_k\) identical parallel, non-preemptive servers
- a FIFO **waiting** buffer of size \(B_k\)

At any time, at most \(N_k\) requests can be in service and at most \(B_k\) can be waiting.

Therefore, the total in-stage capacity is:

\[
N_k + B_k.
\]

---

## 3. Admission and drop (stage 1 only)

A request arriving to stage 1 at time \(t\) is **admitted** iff, after processing all completions at time \(t\), either:

- at least one stage-1 server is idle, or
- the stage-1 waiting buffer has a free slot.

Otherwise, the request is **dropped immediately** at time \(t\).

> Once admitted, a request is never dropped in our case studies.

This definition matches the paper-level metric:

\[
DROP = \frac{D}{A},
\]

where \(A\) is the number of arrivals in the measurement window and \(D\) the number of dropped arrivals.

---

## 4. Blocking-after-service transfer \(1 \rightarrow 2\)

After completing service at stage 1, a request attempts to enter stage 2.

If the current stage-2 in-stage occupancy is strictly less than \(N_2+B_2\), the request enters stage 2 (joining its FIFO queue).

Otherwise, the request **blocks after service**:

- the stage-1 server remains occupied until the transfer succeeds
- blocked transfers are released only when a stage-2 completion frees capacity
- blocked requests are transferred in FIFO order of their stage-1 completion time

---

## 5. Edge-to-cloud extension: network transfer stage

In the edge-to-cloud case study, after completing stage-1 service a request attempts to enter a **network transfer stage** with:

- deterministic delay \(d_{net}\)
- finite in-flight capacity \(B_{net}\)

If fewer than \(B_{net}\) requests are currently in-flight, the request enters the network stage and completes after \(d_{net}\) time units.

If the network in-flight capacity is saturated, the request blocks **before** entering the network stage:

- the stage-1 server remains occupied until a transfer completion frees a slot
- blocked transfers are admitted in FIFO order of their stage-1 completion time

The end-to-end response time \(RT\) includes any waiting for network admission, the deterministic delay \(d_{net}\), and any downstream waiting due to blocking-after-service.

---

## 6. Determinism and reproducibility

All deterministic conventions (tie-breaking, event ordering, and stream separation) are fixed by this contract to support reproducible results across runs and implementations.
