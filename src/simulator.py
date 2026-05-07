"""
SimPy simulator for the two-stage pipeline used in the paper.

Requirements/Assumptions:
- Base time unit: All times are represented as floats in milliseconds (ms).
- Service times: exponential with given means (ms).
- Arrivals: Poisson with (i) steady rate or (ii) deterministic burst windows with a Poisson rate switch.
- Admission / drops: stage 1 only.
- Finite buffers: each stage k has N_k servers and a FIFO waiting buffer B_k.
  Total in-stage capacity is N_k + B_k.
- No-bypass FIFO dispatching and deterministic tie-breaking:
  - FIFO queue at each stage.
  - If multiple servers become free at the same timestamp, assign lowest server index first.
- Blocking-after-service transfer 1->2.
- Edge domain adds a network transfer stage between stage 1 and stage 2:
  - finite in-flight capacity B_net
  - deterministic delay d_net (ms)
  - blocking when saturated.
- Event ordering at the same timestamp MUST be deterministic:
  (1) service completions; (2) arrivals; (3) admission/drop decisions.

This simulator is re-used:
1) during offline induction (Phase 1→5) to evaluate candidates (TEXP)
2) after Phase 6 to evaluate operational families (OPER)
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import simpy


# -----------------------------
# Deterministic priority events
# -----------------------------

PRIO_COMPLETION = 0
PRIO_ARRIVAL = 1
PRIO_ADMISSION = 2


class PrioTimeout(simpy.events.Event):
    """
    A Timeout-like event that is scheduled with an explicit priority.

    SimPy schedules events by (time, priority, event_id). The public Timeout does not expose priority.
    We use this for deterministic same-timestamp ordering required by the contract.
    """

    def __init__(self, env: simpy.Environment, delay: float, priority: int, value=None):
        super().__init__(env)
        self._ok = True
        self._value = value
        if delay < 0:
            raise ValueError("Negative delay")
        env.schedule(self, priority=priority, delay=delay)

    def _desc(self) -> str:  # pragma: no cover
        return f"PrioTimeout({self._value})"


# -----------------------------
# Configuration and descriptors
# -----------------------------

@dataclass(frozen=True)
class Descriptor:
    """
    Descriptor x = (rho, b)
      rho: load factor
      b: 0 steady, 1 burst windows
    """
    desc_id: str
    rho: float
    burst_flag: int


@dataclass(frozen=True)
class SimConfig:
    domain: str  # "dc" | "edge"
    horizon_ms: float
    warmup_ms: float
    replications_R: int

    # Service means (ms)
    m1_ms: float
    m2_ms: float

    # Workload constants
    N2_ref: int
    rho_grid: List[float]
    burst_flags: List[int]
    P_ms: float
    W_ms: float
    F: float

    # Edge extras
    net_delay_ms: float = 0.0
    net_buffer_Bnet: int = 0


@dataclass(frozen=True)
class TwinParams:
    twin_id: str
    N1: int
    N2: int
    B1: int
    B2: int


# -----------------------------
# Seeding (contract)
# -----------------------------

def _seed_u64(*parts: object) -> int:
    """
    Deterministic 64-bit seed derived from SHA-256 of the tuple of parts.
    Contract: "take the first 8 bytes as an unsigned 64-bit seed".
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    digest = h.digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


# -----------------------------
# Workload rates (requests/ms)
# -----------------------------

def _lambda_base_ms(rho: float, N2_ref: int, m2_ms: float) -> float:
    """
    Base arrival rate lambda_base = rho * C_ref where C_ref = N2_ref * (1/m2).
    (All in ms units: 1/m2 is requests/ms per server.)
    """
    if m2_ms <= 0:
        return 0.0
    C_ref = float(N2_ref) * (1.0 / float(m2_ms))
    return float(rho) * C_ref


def _burst_rates(lambda_base: float, P_ms: float, W_ms: float, F: float) -> Tuple[float, float]:
    """
    Workload calibration:
      duty d = W/P
      lambda_low = lambda_base / ((1-d) + d*F)
      lambda_high = F * lambda_low
    """
    if P_ms <= 0 or W_ms <= 0 or F <= 0:
        return lambda_base, lambda_base
    d = float(W_ms) / float(P_ms)
    denom = (1.0 - d) + d * float(F)
    if denom <= 0:
        return lambda_base, lambda_base
    lam_low = float(lambda_base) / denom
    lam_high = float(F) * lam_low
    return lam_low, lam_high


def _rate_at(t_ms: float, lambda_base: float, burst_flag: int, P_ms: float, W_ms: float, F: float) -> float:
    """
    Rate at time t (ms).
    - If burst_flag=0: constant lambda_base.
    - If burst_flag=1: deterministic burst windows.
      We assume that each period P starts with the HIGH window of length W (then LOW for P-W).
      (We fix the phase offset deterministically at t=0.)
    """
    if burst_flag == 0:
        return float(lambda_base)
    lam_low, lam_high = _burst_rates(lambda_base, P_ms, W_ms, F)
    if P_ms <= 0:
        return float(lambda_base)
    phase = float(t_ms) % float(P_ms)
    return lam_high if phase < float(W_ms) else lam_low


def _next_boundary_ms(t_ms: float, P_ms: float, W_ms: float) -> float:
    """
    Next rate-change boundary strictly after time t for the burst schedule.
    """
    if P_ms <= 0:
        return math.inf
    phase = float(t_ms) % float(P_ms)
    # boundaries at phase=W and phase=P(=0)
    if phase < float(W_ms):
        return float(t_ms) + (float(W_ms) - phase)
    return float(t_ms) + (float(P_ms) - phase)


def _sample_next_arrival_dt(now_ms: float, lambda_base: float, burst_flag: int,
                            P_ms: float, W_ms: float, F: float, rng: np.random.Generator) -> float:
    """
    Sample the next inter-arrival time with a piecewise-constant Poisson rate.
    Uses the "boundary resampling" approach:
      - sample Exp(rate) under current rate
      - if it crosses a boundary, advance to boundary and resample under new rate
    """
    t = float(now_ms)
    while True:
        r = _rate_at(t, lambda_base, burst_flag, P_ms, W_ms, F)
        if r <= 0:
            return math.inf
        dt = float(rng.exponential(1.0 / r))
        if burst_flag == 0:
            return dt
        b = _next_boundary_ms(t, P_ms, W_ms)
        if t + dt < b:
            return dt
        # Crossed boundary: move to boundary and resample
        t = b


# -----------------------------
# Core pipeline model
# -----------------------------

class _Stage:
    """
    A single stage with:
      - N identical non-preemptive servers
      - FIFO waiting buffer of size B
      - deterministic tie-breaking (lowest server index first)
      - optional "post-service blocking": a server is not released until the caller releases it.
    """

    def __init__(self, env: simpy.Environment, name: str, N: int, B: int, service_mean_ms: float,
                 rng_service: np.random.Generator):
        self.env = env
        self.name = name
        self.N = int(N)
        self.B = int(B)
        self.service_mean_ms = float(service_mean_ms)
        self.rng_service = rng_service

        # Queue of waiting jobs (FIFO)
        self.wait_q: List["_Job"] = []

        # Server state: None means idle, otherwise holds job
        self.server_job: List[Optional["_Job"]] = [None for _ in range(self.N)]

        # Used to wake up the dispatcher when something changes
        self._changed = simpy.Event(env)

    def total_in_stage(self) -> int:
        in_service = sum(1 for j in self.server_job if j is not None)
        return in_service + len(self.wait_q)

    def can_accept(self) -> bool:
        return self.total_in_stage() < (self.N + self.B)

    def accept(self, job: "_Job") -> None:
        """
        Enqueue a job (assumes can_accept() is True).
        """
        # If there is an idle server, assign immediately with lowest index.
        for idx in range(self.N):
            if self.server_job[idx] is None:
                self.server_job[idx] = job
                self.env.process(self._run_on_server(idx, job))
                self._trigger_changed()
                return

        # Otherwise, push into waiting buffer (FIFO). Capacity already validated.
        self.wait_q.append(job)
        self._trigger_changed()

    def release_server(self, idx: int) -> None:
        """
        Release a server (called by the job after it successfully transfers downstream, to implement
        blocking-after-service).
        """
        self.server_job[idx] = None
        self._trigger_changed()

    def _trigger_changed(self) -> None:
        if not self._changed.triggered:
            self._changed.succeed()
        self._changed = simpy.Event(self.env)

    def _sample_service_time(self) -> float:
        if self.service_mean_ms <= 0:
            return 0.0
        return float(self.rng_service.exponential(self.service_mean_ms))

    def _run_on_server(self, idx: int, job: "_Job") -> simpy.events.Event:
        """
        Process service for one job on one server. After service completion, the job must call
        job.on_stage_complete(...) which may block (transfer semantics).
        """
        # Service completion must be priority 0 (before arrivals).
        st = self._sample_service_time()
        yield PrioTimeout(self.env, st, PRIO_COMPLETION)

        # Now invoke stage-completion handler (may block until downstream capacity).
        yield from job.on_stage_complete(self, idx)

        # Server release happens inside job.on_stage_complete when transfer completes.
        # After release, dispatcher will pick next waiting jobs.


class _Network:
    """
    Edge-only network transfer stage with:
      - finite in-flight capacity B_net
      - deterministic delay d_net
      - blocking when saturated
    """

    def __init__(self, env: simpy.Environment, B_net: int, d_net_ms: float):
        self.env = env
        self.B_net = int(B_net)
        self.d_net_ms = float(d_net_ms)

        # FIFO queue of jobs waiting to enter the network when in-flight is full
        self.wait_q: List["_Job"] = []
        self.in_flight: int = 0

        # Waker
        self._changed = simpy.Event(env)

    def can_enter(self) -> bool:
        return self.in_flight < self.B_net

    def request_enter(self, job: "_Job") -> simpy.events.Event:
        """
        Return an event that completes when the job is admitted into the network (i.e., allocated
        an in-flight slot). FIFO if waiting.
        """
        if self.B_net <= 0:
            # No network capacity: treat as always saturated => immediate admission makes no sense.
            # We interpret B_net<=0 as "no network stage"; handled by cfg.domain.
            ev = simpy.Event(self.env)
            ev.succeed()
            return ev

        if self.can_enter() and not self.wait_q:
            self.in_flight += 1
            ev = simpy.Event(self.env)
            ev.succeed()
            self._trigger_changed()
            return ev

        # Otherwise enqueue and wait
        ev = simpy.Event(self.env)
        job._net_admit_event = ev
        self.wait_q.append(job)
        return ev

    def _trigger_changed(self) -> None:
        if not self._changed.triggered:
            self._changed.succeed()
        self._changed = simpy.Event(self.env)

    def release_in_flight(self) -> None:
        self.in_flight -= 1
        if self.in_flight < 0:
            self.in_flight = 0

        # Admit waiting jobs in FIFO order until full
        while self.wait_q and self.can_enter():
            j = self.wait_q.pop(0)
            self.in_flight += 1
            if j._net_admit_event is not None and (not j._net_admit_event.triggered):
                j._net_admit_event.succeed()
            j._net_admit_event = None

        self._trigger_changed()

    def delay(self) -> simpy.events.Event:
        # Network delay completion must be treated as a "service completion" (priority 0),
        # because it is a stage completion event.
        return PrioTimeout(self.env, self.d_net_ms, PRIO_COMPLETION)


@dataclass
class _Job:
    """
    One request flowing through the pipeline.
    """
    req_id: int
    arrival_time_ms: float
    admitted_time_ms: Optional[float] = None
    stage1_start_ms: Optional[float] = None
    stage2_start_ms: Optional[float] = None
    completion_time_ms: Optional[float] = None

    # internal for edge network
    _net_admit_event: Optional[simpy.Event] = None

    def on_stage_complete(self, stage: _Stage, server_idx: int):
        """
        Called by a stage server after service time completes.
        Implements transfer semantics and blocking-after-service.
        """
        raise NotImplementedError


class _PipelineModel:
    def __init__(self, env: simpy.Environment, cfg: SimConfig, twin: TwinParams,
                 rng_s1: np.random.Generator, rng_s2: np.random.Generator):
        self.env = env
        self.cfg = cfg
        self.twin = twin

        self.stage1 = _Stage(env, "stage1", twin.N1, twin.B1, cfg.m1_ms, rng_s1)
        self.stage2 = _Stage(env, "stage2", twin.N2, twin.B2, cfg.m2_ms, rng_s2)

        self.net: Optional[_Network] = None
        if cfg.domain == "edge":
            self.net = _Network(env, cfg.net_buffer_Bnet, cfg.net_delay_ms)

        # Metrics (computed on [Tw, T])
        self.arrivals_meas = 0
        self.drops_meas = 0
        self.rts_meas: List[float] = []

        self._next_req_id = 0

    def make_job(self, now_ms: float) -> _Job:
        rid = self._next_req_id
        self._next_req_id += 1
        model = self

        class JobImpl(_Job):
            def on_stage_complete(self, stage: _Stage, server_idx: int):
                # stage 1 -> stage 2 transfer (with optional network)
                if stage is model.stage1:
                    if model.net is None:
                        # Direct blocking-after-service to stage2 capacity
                        yield from model._transfer_to_stage2(self, stage, server_idx)
                        return
                    # Edge: first enter network (may block)
                    yield model.net.request_enter(self)
                    # Network delay
                    yield model.net.delay()
                    # After delay, attempt to enter stage2 (may block)
                    yield from model._transfer_to_stage2(self, stage, server_idx, via_network=True)
                    return

                # stage 2 completion -> sink
                if stage is model.stage2:
                    self.completion_time_ms = float(model.env.now)
                    # Record RT if within measurement window
                    if self.arrival_time_ms >= float(model.cfg.warmup_ms):
                        rt = float(self.completion_time_ms) - float(self.arrival_time_ms)
                        model.rts_meas.append(rt)
                    # Release stage2 server immediately
                    stage.release_server(server_idx)
                    return

                raise RuntimeError("Unknown stage")

        return JobImpl(req_id=rid, arrival_time_ms=float(now_ms))

    def _transfer_to_stage2(self, job: _Job, stage1: _Stage, s1_idx: int, via_network: bool = False):
        """
        Transfer logic to stage 2 under blocking-after-service (Contract).
        When saturated, the job blocks (holding its current resource) until capacity opens.
        """
        # If via_network, the job holds a network in-flight slot until stage2 admits it.
        while True:
            if self.stage2.can_accept():
                # Admit into stage2 at current time (deterministic)
                self.stage2.accept(job)
                # Release upstream resource:
                # - always release stage1 server once admitted downstream
                stage1.release_server(s1_idx)
                # - if edge: release network in-flight slot after stage2 admission
                if via_network and self.net is not None:
                    self.net.release_in_flight()
                return

            # Wait until something changes in stage2 (a completion frees capacity)
            # This is blocking-after-service: keep holding resources.
            yield self.stage2._changed

    # Admission is processed as a separate priority-2 event.
    # IMPORTANT: In SimPy, a "process" must be a generator. The admission procedure is
    # triggered by a scheduled event callback (priority=PRIO_ADMISSION) and then executed
    # as a process.
    def admission_event(self, job: _Job):
        """Admission decision for one job (SimPy process)."""
        # Admission decisions must be after all arrivals at same timestamp (priority 2).
        # Here we are already in the admission event callback at that time.
        now_ms = float(self.env.now)
        if now_ms >= float(self.cfg.warmup_ms):
            self.arrivals_meas += 1

        if self.stage1.can_accept():
            job.admitted_time_ms = now_ms
            self.stage1.accept(job)
        else:
            if now_ms >= float(self.cfg.warmup_ms):
                self.drops_meas += 1

        # Make this a generator without changing simulation time.
        # (SimPy requires processes to be generators; returning immediately is fine.)
        if False:  # pragma: no cover
            yield self.env.timeout(0)


# -----------------------------
# Simulation runner
# -----------------------------

def simulate_once(
    run_id: str,
    cfg: SimConfig,
    twin: TwinParams,
    desc: Descriptor,
    replication_index: int,
) -> Dict[str, float]:
    """
    Run one replication and return primitive indicators:
      - rt_p99_ms
      - drop_rate
    """
    env = simpy.Environment()

    # Independent random streams (Contract)
    rng_arr = np.random.default_rng(_seed_u64(run_id, desc.desc_id, twin.twin_id, replication_index, "arrivals"))
    rng_s1 = np.random.default_rng(_seed_u64(run_id, desc.desc_id, twin.twin_id, replication_index, "service_s1"))
    rng_s2 = np.random.default_rng(_seed_u64(run_id, desc.desc_id, twin.twin_id, replication_index, "service_s2"))

    model = _PipelineModel(env, cfg, twin, rng_s1=rng_s1, rng_s2=rng_s2)

    # Arrival process: schedule arrivals with PRIO_ARRIVAL, and schedule admission with PRIO_ADMISSION.
    lambda_base = _lambda_base_ms(desc.rho, cfg.N2_ref, cfg.m2_ms)

    def arrival_process():
        t = 0.0
        while True:
            dt = _sample_next_arrival_dt(t, lambda_base, desc.burst_flag, cfg.P_ms, cfg.W_ms, cfg.F, rng_arr)
            if math.isinf(dt):
                return
            t = t + dt
            if t > float(cfg.horizon_ms):
                return

            # Wait until arrival time (priority 1)
            yield PrioTimeout(env, dt, PRIO_ARRIVAL)

            now_ms = float(env.now)
            job = model.make_job(now_ms)

            # Admission decisions must happen at the same timestamp but after
            # all arrivals. We model this as a zero-delay timeout with a lower
            # priority (larger priority value means later within the same time).
            yield PrioTimeout(env, 0, PRIO_ADMISSION)
            # Perform admission as a SimPy process (generator).
            yield from model.admission_event(job)

    env.process(arrival_process())

    # Run
    env.run(until=float(cfg.horizon_ms))

    # Metrics
    A = model.arrivals_meas
    D = model.drops_meas
    drop_rate = (float(D) / float(A)) if A > 0 else 0.0

    rts = model.rts_meas
    if not rts:
        rt_p50 = 0.0
        rt_p95 = 0.0
        rt_p99 = 0.0
        rt_max = 0.0
    else:
        rts_sorted = sorted(rts)

        def _nearest_rank(p: float) -> float:
            idx = int(math.ceil(p * len(rts_sorted))) - 1
            idx = max(0, min(idx, len(rts_sorted) - 1))
            return float(rts_sorted[idx])

        rt_p50 = _nearest_rank(0.50)
        rt_p95 = _nearest_rank(0.95)
        rt_p99 = _nearest_rank(0.99)
        rt_max = float(rts_sorted[-1])

    rt_mean = float(sum(rts) / len(rts)) if rts else 0.0
    return {
"rt_mean": rt_mean,
        "rt_p50": rt_p50,
        "rt_p95": rt_p95,
        "rt_p99": rt_p99,
        "rt_max": rt_max,
        "drop": drop_rate,
        "completed": int(len(rts)),
        "arrivals": int(A),
        "drops": int(D),

    }

def simulate_replication(cfg_dict: Dict, twin_dict: Dict, rho: float, burst: int, seed: int = 0) -> Dict[str, float]:
    """
    Backward-compatible wrapper expected by the evaluation pipeline (phase3_eval.py).

    This wrapper accepts either:
      - a flattened config dict (legacy), OR
      - the full YAML-loaded experiment dict (recommended), with a "sim" section.

    The simulator core uses deterministic stream splitting based on:
      (run_id, desc_id, twin_id, replication_index, stream_name)

    We map:
      # NOTE: do NOT use per-run environment-derived IDs (e.g., DTWIN_RUN_ID)
      # in the seed material, otherwise identical configurations will produce
      # different simulation streams across runs.
      run_id = str(cfg_dict.get("run_id", "run"))
      replication_index = int(seed)
      desc_id = f"rho{rho}_b{burst}"

    Returns keys expected by the evaluation pipeline:
      rt_mean, rt_p99, drop, completed, arrivals, drops
    """
    # Support nested YAML structure: cfg_dict["sim"] contains simulation parameters.
    sim = cfg_dict.get("sim", cfg_dict)

    # Domain is at top-level in our YAML.
    domain = str(cfg_dict.get("domain", sim.get("domain", "dc")))

    # Service means in ms
    svc = sim.get("service_means_ms", {})
    m1_ms = float(sim.get("m1_ms", svc.get("m1", 0.0)))
    m2_ms = float(sim.get("m2_ms", svc.get("m2", 0.0)))

    # Workload descriptors
    wl = sim.get("workload", {})
    rho_grid = wl.get("rho_grid", sim.get("rho_grid", [rho]))
    burst_flags = wl.get("burst_flags", sim.get("burst_flags", [burst]))
    burst_cfg = wl.get("burst", {})
    P_ms = float(sim.get("P_ms", burst_cfg.get("P_ms", 0.0)))
    W_ms = float(sim.get("W_ms", burst_cfg.get("W_ms", 0.0)))
    F = float(sim.get("F", burst_cfg.get("F", 1.0)))

    # Reference capacity
    ref = sim.get("reference_capacity", {})
    N2_ref = int(sim.get("N2_ref", ref.get("N2_ref", 1)))

    # Edge network
    net = sim.get("network", {})
    net_delay_ms = float(sim.get("net_delay_ms", net.get("delay_ms", 0.0)))
    net_buffer_Bnet = int(sim.get("net_buffer_Bnet", net.get("buffer_Bnet", 0)))

    cfg = SimConfig(
        domain=domain,
        horizon_ms=float(sim.get("horizon_ms", cfg_dict.get("horizon_ms", 0.0))),
        warmup_ms=float(sim.get("warmup_ms", cfg_dict.get("warmup_ms", 0.0))),
        replications_R=int(sim.get("replications_R", sim.get("replications", cfg_dict.get("replications_R", 1)))),
        m1_ms=m1_ms,
        m2_ms=m2_ms,
        N2_ref=N2_ref,
        rho_grid=[float(x) for x in rho_grid],
        burst_flags=[int(x) for x in burst_flags],
        P_ms=P_ms,
        W_ms=W_ms,
        F=F,
        net_delay_ms=net_delay_ms,
        net_buffer_Bnet=net_buffer_Bnet,
    )

    twin = TwinParams(
        twin_id=str(twin_dict.get("twin_id", twin_dict.get("id", "t"))),
        N1=int(twin_dict["N1"]),
        N2=int(twin_dict["N2"]),
        B1=int(twin_dict["B1"]),
        B2=int(twin_dict["B2"]),
    )

    # Determinism: ignore DTWIN_RUN_ID. run_all.py sets it to a timestamped value
    # for output folder naming, but that must NOT affect random streams.
    run_id = str(cfg_dict.get("run_id", "run"))
    desc_id = f"rho{rho}_b{burst}"

    # Call the simulator core for a single replication.
    out = simulate_once(run_id, cfg, twin, Descriptor(desc_id=desc_id, rho=float(rho), burst_flag=int(burst)), replication_index=int(seed))
# Normalize keys for the evaluation pipeline.
    rt_mean = float(out.get("rt_mean", 0.0))
    rt_p50 = float(out.get("rt_p50", 0.0))
    rt_p95 = float(out.get("rt_p95", 0.0))
    rt_p99 = float(out.get("rt_p99", 0.0))
    rt_max = float(out.get("rt_max", 0.0))
    drop = float(out.get("drop", 0.0))
    completed = int(out.get("completed", 0))
    arrivals = int(out.get("arrivals", 0))
    drops = int(out.get("drops", 0))

    return {
        "rt_mean": rt_mean,
        "rt_p50": rt_p50,
        "rt_p95": rt_p95,
        "rt_p99": rt_p99,
        "rt_max": rt_max,
        "drop": drop,
        "completed": completed,
        "arrivals": arrivals,
        "drops": drops,
    }
