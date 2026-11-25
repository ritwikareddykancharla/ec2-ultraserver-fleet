"""Collective Wait-For Straggler/Deadlock Detector (CUSUM) & Contiguous Sub-Ring Gang Scheduler.

1. Page-Hinkley / tabular CUSUM change-point detector + directed Wait-For graph analyzer
   (sink root-blocker isolation & Tarjan DFS cycle detection) on multi-rank collective traces.
2. Topology-preserving Buddy Sub-Ring allocator + EASY Backfill gang scheduler routing
   multi-node jobs onto contiguous healthy rack sub-rings around cordoned hosts with zero
   cross-spine ring fragmentation.
"""
from __future__ import annotations

import math
import random
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# ==============================================================================
# PART 1: CUSUM Change-Point + Directed Wait-For Graph Straggler/Deadlock Finder
# ==============================================================================


def generate_collective_trace(
    preset: str = "rank11_nic_straggler",
    num_ranks: int = 16,
    num_steps: int = 44,
    culprit_rank: Optional[int] = None,
    seed: int = 9101,
) -> Dict[str, Any]:
    """Generate a realistic multi-rank collective trace with step latencies and Wait-For edges."""
    rng = random.Random(seed)
    base_ms = 42.0
    noise_sd = 1.8

    if preset == "dual_rank_deadlock_cycle":
        cycle_ranks = [3, 7, 12] if num_ranks >= 13 else [0, 1, 2]
        rank_series: Dict[int, List[float]] = {}
        for r in range(num_ranks):
            series = []
            for s in range(num_steps):
                if r in cycle_ranks and s >= 24:
                    series.append(round(base_ms + 95.0 + rng.gauss(0, 1.5), 2))
                else:
                    series.append(round(base_ms + rng.gauss(0, noise_sd), 2))
            rank_series[r] = series

        wait_edges: List[Dict[str, Any]] = []
        for idx, u in enumerate(cycle_ranks):
            v = cycle_ranks[(idx + 1) % len(cycle_ranks)]
            wait_edges.append({"waiter": u, "blocker": v, "op": "AllToAll_TagSync", "wait_ms": 980.0})
        # Remaining ranks block downstream on cycle_ranks[0]
        for r in range(num_ranks):
            if r not in cycle_ranks:
                wait_edges.append(
                    {"waiter": r, "blocker": cycle_ranks[0], "op": "RingAllReduce_Barrier", "wait_ms": 940.0}
                )
        return {
            "trace_id": "trc-deadlock-cycle-03",
            "title": "3-Rank Collective Tag Deadlock Cycle (Ranks 3 -> 7 -> 12 -> 3)",
            "mode": "deadlock_cycle",
            "num_ranks": num_ranks,
            "num_steps": num_steps,
            "timeout_step": 40,
            "ground_truth_culprits": cycle_ranks,
            "rank_step_ms": {str(k): v for k, v in rank_series.items()},
            "wait_for_edges": wait_edges,
        }

    # Single-rank progressive straggler (EFA link CRC retransmit or thermal clock cap)
    if culprit_rank is None:
        culprit_rank = 5 if preset == "rank05_thermal_throttle" else min(11, num_ranks - 1)
    onset_step = 14 if preset == "rank05_thermal_throttle" else 18
    drift_slope = 3.6 if preset == "rank05_thermal_throttle" else 4.8

    rank_series = {}
    for r in range(num_ranks):
        series = []
        for s in range(num_steps):
            val = base_ms + rng.gauss(0, noise_sd)
            if r == culprit_rank and s >= onset_step:
                val += min(88.0, (s - onset_step + 1) * drift_slope)
            series.append(round(val, 2))
        rank_series[r] = series

    # Ring AllReduce wait-for chain terminating at culprit_rank (out-degree 0)
    wait_edges = []
    for offset in range(1, num_ranks):
        waiter = (culprit_rank - offset) % num_ranks
        blocker = (culprit_rank - offset + 1) % num_ranks
        wait_edges.append(
            {
                "waiter": waiter,
                "blocker": blocker,
                "op": "RingAllReduce_SendRecv",
                "wait_ms": round(max(40.0, 860.0 - offset * 18.0), 1),
            }
        )

    title = (
        f"Rank {culprit_rank:02d} Thermal Clock Cap Straggler (16-Rank Hybrid Ring)"
        if preset == "rank05_thermal_throttle"
        else f"Rank {culprit_rank:02d} EFA Retransmit Straggler (16-Rank Ring AllReduce)"
    )
    return {
        "trace_id": f"trc-{preset}",
        "title": title,
        "mode": "straggler",
        "num_ranks": num_ranks,
        "num_steps": num_steps,
        "onset_step": onset_step,
        "timeout_step": 40,
        "ground_truth_culprits": [culprit_rank],
        "rank_step_ms": {str(k): v for k, v in rank_series.items()},
        "wait_for_edges": wait_edges,
    }


def _detect_cycles_tarjan(num_ranks: int, adj: Dict[int, List[int]]) -> List[List[int]]:
    """Find non-trivial directed cycles via Tarjan's strongly connected components."""
    index_counter = [0]
    indices: Dict[int, int] = {}
    lowlink: Dict[int, int] = {}
    stack: List[int] = []
    on_stack: Set[int] = set()
    cycles: List[List[int]] = []

    def strongconnect(v: int) -> None:
        indices[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in adj.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            scc: List[int] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == v:
                    break
            if len(scc) > 1 or (len(scc) == 1 and scc[0] in adj.get(scc[0], [])):
                cycles.append(sorted(scc))

    for node in range(num_ranks):
        if node not in indices:
            strongconnect(node)
    return cycles


def analyze_collective_trace(
    trace: Dict[str, Any],
    cusum_slack_k: float = 0.5,
    cusum_threshold_h: float = 4.5,
) -> Dict[str, Any]:
    """Run tabular upper CUSUM + directed Wait-For graph analysis to isolate root culprit ranks."""
    t0 = time.perf_counter()
    num_ranks = int(trace["num_ranks"])
    timeout_step = int(trace.get("timeout_step", trace["num_steps"] - 1))
    rank_step_ms: Dict[int, List[float]] = {
        int(k): [float(x) for x in v] for k, v in trace["rank_step_ms"].items()
    }

    # Estimate fleet baseline mu0, sigma0 across warmup window [0..9] over all ranks (robust median/MAD)
    warmup_vals: List[float] = []
    for r in range(num_ranks):
        warmup_vals.extend(rank_step_ms[r][: min(10, len(rank_step_ms[r]))])
    mu0 = statistics.median(warmup_vals)
    mad = statistics.median([abs(x - mu0) for x in warmup_vals])
    sigma0 = max(1.0, mad * 1.4826)

    per_rank_cusum: List[Dict[str, Any]] = []
    flagged_cusum_ranks: List[int] = []

    for r in range(num_ranks):
        s_pos = 0.0
        peak_s = 0.0
        alarm_step: Optional[int] = None
        for step_idx, val in enumerate(rank_step_ms[r]):
            z = (val - mu0) / sigma0
            s_pos = max(0.0, s_pos + z - cusum_slack_k)
            if s_pos > peak_s:
                peak_s = s_pos
            if alarm_step is None and s_pos >= cusum_threshold_h:
                alarm_step = step_idx
        if alarm_step is not None:
            flagged_cusum_ranks.append(r)
        per_rank_cusum.append(
            {
                "rank": r,
                "mean_step_ms": round(statistics.mean(rank_step_ms[r]), 2),
                "tail_step_ms": round(statistics.mean(rank_step_ms[r][-6:]), 2),
                "peak_cusum": round(peak_s, 2),
                "change_point_step": alarm_step,
                "lead_steps_before_timeout": (timeout_step - alarm_step) if alarm_step is not None else 0,
            }
        )

    # Build directed Wait-For graph: waiter -> blocker
    adj: Dict[int, List[int]] = {r: [] for r in range(num_ranks)}
    rev_adj: Dict[int, List[int]] = {r: [] for r in range(num_ranks)}
    for edge in trace.get("wait_for_edges", []):
        u = int(edge["waiter"])
        v = int(edge["blocker"])
        adj[u].append(v)
        rev_adj[v].append(u)

    cycles = _detect_cycles_tarjan(num_ranks, adj)

    # Compute transitive upstream waiters blocked by each rank
    blocked_dependents: Dict[int, int] = {}
    for r in range(num_ranks):
        visited: Set[int] = set()
        stack = list(rev_adj[r])
        while stack:
            cur = stack.pop()
            if cur not in visited and cur != r:
                visited.add(cur)
                stack.extend(rev_adj[cur])
        blocked_dependents[r] = len(visited)

    if cycles:
        diagnosis_type = "COLLECTIVE_DEADLOCK_CYCLE"
        isolated_culprits = cycles[0]
        lead_steps = max(
            (per_rank_cusum[r]["lead_steps_before_timeout"] for r in isolated_culprits),
            default=0,
        )
    else:
        diagnosis_type = "SINGLE_RANK_STRAGGLER"
        # Sink nodes (out-degree == 0) weighted by transitive waiters + peak CUSUM score
        sinks = [r for r in range(num_ranks) if len(adj[r]) == 0 and blocked_dependents[r] > 0]
        candidates = sinks if sinks else list(range(num_ranks))
        best_rank = max(
            candidates,
            key=lambda r: (blocked_dependents[r], per_rank_cusum[r]["peak_cusum"]),
        )
        isolated_culprits = [best_rank]
        lead_steps = per_rank_cusum[best_rank]["lead_steps_before_timeout"]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    noisy_raw_alerts = num_ranks
    alert_compression_ratio = f"{noisy_raw_alerts}:{len(isolated_culprits)}"

    for item in per_rank_cusum:
        r = item["rank"]
        item["out_degree"] = len(adj[r])
        item["transitive_waiters"] = blocked_dependents[r]
        item["is_root_culprit"] = r in isolated_culprits

    return {
        "trace_id": trace.get("trace_id", "custom-trace"),
        "title": trace.get("title", "Collective Trace Analysis"),
        "diagnosis_type": diagnosis_type,
        "isolated_culprits": isolated_culprits,
        "deadlock_cycles": cycles,
        "flagged_cusum_ranks": flagged_cusum_ranks,
        "lead_steps_before_timeout": lead_steps,
        "noisy_timeout_ranks_suppressed": num_ranks - len(isolated_culprits),
        "alert_compression": alert_compression_ratio,
        "baseline_mu_ms": round(mu0, 2),
        "baseline_sigma_ms": round(sigma0, 2),
        "analysis_ms": round(elapsed_ms, 3),
        "rank_diagnostics": per_rank_cusum,
    }


def run_synthetic_straggler_benchmark(
    num_traces: int = 100,
    num_ranks: int = 16,
    num_steps: int = 44,
    seed: int = 20261125,
) -> Dict[str, Any]:
    """Benchmark CUSUM + Wait-For culprit isolation across 100 synthetic collective traces."""
    rng = random.Random(seed)
    exact_matches = 0
    false_peer_blames = 0
    lead_steps_list: List[int] = []
    latencies_ms: List[float] = []

    for idx in range(num_traces):
        if idx % 5 == 4:
            # 20% deadlock cycle traces across random 3-rank cycles
            cycle_ranks = sorted(rng.sample(range(num_ranks), 3))
            tr = generate_collective_trace(
                preset="dual_rank_deadlock_cycle",
                num_ranks=num_ranks,
                num_steps=num_steps,
                seed=seed + idx * 17,
            )
            # Rewire cycle to random chosen 3-rank cycle
            wait_edges = []
            for c_i, u in enumerate(cycle_ranks):
                v = cycle_ranks[(c_i + 1) % 3]
                wait_edges.append({"waiter": u, "blocker": v, "op": "AllToAll_TagSync", "wait_ms": 960.0})
                for step_i in range(24, num_steps):
                    tr["rank_step_ms"][str(u)][step_i] = round(138.0 + rng.gauss(0, 1.5), 2)
            for r in range(num_ranks):
                if r not in cycle_ranks:
                    tr["rank_step_ms"][str(r)] = [round(42.0 + rng.gauss(0, 1.8), 2) for _ in range(num_steps)]
                    wait_edges.append({"waiter": r, "blocker": cycle_ranks[0], "op": "Barrier", "wait_ms": 910.0})
            tr["wait_for_edges"] = wait_edges
            tr["ground_truth_culprits"] = cycle_ranks
        else:
            culprit = rng.randrange(num_ranks)
            preset = "rank05_thermal_throttle" if idx % 2 == 1 else "rank11_nic_straggler"
            tr = generate_collective_trace(
                preset=preset,
                num_ranks=num_ranks,
                num_steps=num_steps,
                culprit_rank=culprit,
                seed=seed + idx * 17,
            )

        res = analyze_collective_trace(tr)
        latencies_ms.append(res["analysis_ms"])
        truth = sorted(tr["ground_truth_culprits"])
        pred = sorted(res["isolated_culprits"])
        if pred == truth:
            exact_matches += 1
        false_peer_blames += len(set(pred) - set(truth))
        if res["lead_steps_before_timeout"] > 0:
            lead_steps_list.append(res["lead_steps_before_timeout"])

    lat_sorted = sorted(latencies_ms)
    p50 = lat_sorted[len(lat_sorted) // 2]
    p95 = lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95))]

    return {
        "num_traces": num_traces,
        "num_ranks_per_trace": num_ranks,
        "exact_culprit_matches": exact_matches,
        "culprit_accuracy_pct": round((exact_matches / num_traces) * 100.0, 2),
        "false_peer_blames": false_peer_blames,
        "mean_lead_steps_before_timeout": round(statistics.mean(lead_steps_list), 2) if lead_steps_list else 0.0,
        "analysis_p50_ms": round(p50, 3),
        "analysis_p95_ms": round(p95, 3),
    }


# ==============================================================================
# PART 2: Contiguous Buddy Sub-Ring + EASY Backfill Gang Scheduler
# ==============================================================================

DEFAULT_WORKLOAD_PRESETS: Dict[str, List[Dict[str, Any]]] = {
    "mixed_training_and_ci": [
        {"job_id": "job-llm-8n-a", "nodes": 8, "duration": 10, "kind": "8-Node Ring Pretrain"},
        {"job_id": "job-ci-1n-01", "nodes": 1, "duration": 3, "kind": "1-Node Kernel Smoke"},
        {"job_id": "job-llm-8n-b", "nodes": 8, "duration": 10, "kind": "8-Node Ring Pretrain"},
        {"job_id": "job-ft-4n-01", "nodes": 4, "duration": 6, "kind": "4-Node Sub-Ring LoRA"},
        {"job_id": "job-llm-16n-hero", "nodes": 16, "duration": 8, "kind": "16-Node Dual-Rack Gang"},
        {"job_id": "job-eval-2n-01", "nodes": 2, "duration": 4, "kind": "2-Node Eval Pair"},
        {"job_id": "job-eval-2n-02", "nodes": 2, "duration": 4, "kind": "2-Node Eval Pair"},
        {"job_id": "job-ci-1n-02", "nodes": 1, "duration": 3, "kind": "1-Node Kernel Smoke"},
        {"job_id": "job-ci-1n-03", "nodes": 1, "duration": 2, "kind": "1-Node Kernel Smoke"},
        {"job_id": "job-ft-4n-02", "nodes": 4, "duration": 5, "kind": "4-Node Sub-Ring LoRA"},
    ],
    "heavy_gang_contention": [
        {"job_id": "gang-8n-alpha", "nodes": 8, "duration": 12, "kind": "8-Node Dense Ring"},
        {"job_id": "gang-8n-beta", "nodes": 8, "duration": 9, "kind": "8-Node Dense Ring"},
        {"job_id": "gang-16n-gamma", "nodes": 16, "duration": 10, "kind": "16-Node Dual-Rack Gang"},
        {"job_id": "backfill-2n-01", "nodes": 2, "duration": 5, "kind": "2-Node Sub-Ring Bench"},
        {"job_id": "backfill-2n-02", "nodes": 2, "duration": 4, "kind": "2-Node Sub-Ring Bench"},
        {"job_id": "backfill-1n-03", "nodes": 1, "duration": 4, "kind": "1-Node Canary Check"},
        {"job_id": "backfill-1n-04", "nodes": 1, "duration": 3, "kind": "1-Node Canary Check"},
        {"job_id": "gang-4n-delta", "nodes": 4, "duration": 6, "kind": "4-Node Sub-Ring Train"},
        {"job_id": "gang-4n-eps", "nodes": 4, "duration": 6, "kind": "4-Node Sub-Ring Train"},
        {"job_id": "backfill-2n-05", "nodes": 2, "duration": 3, "kind": "2-Node Sub-Ring Bench"},
    ],
}


def _find_buddy_allocation(
    free_mask_by_rack: Dict[str, List[bool]],
    nodes_needed: int,
    rack_size: int = 8,
) -> Optional[List[Tuple[str, int]]]:
    """Find a zero-fragmentation aligned Buddy Sub-Ring allocation on healthy free slots.

    - For nodes_needed in {1, 2, 4, 8}: chooses an aligned block [b*s .. (b+1)*s - 1] inside a single rack,
      preferring racks that are ALREADY partially occupied/cordoned (best-fit buddy) so full 8-node rings
      remain unfragmented for large gang jobs.
    - For nodes_needed == 16: allocates 2 complete healthy 8-node racks (zero partial-rack cross-spine split).
    """
    racks_sorted = sorted(free_mask_by_rack.keys())

    if nodes_needed == 16:
        full_racks = [
            rk for rk in racks_sorted if sum(1 for x in free_mask_by_rack[rk] if x) == rack_size
        ]
        if len(full_racks) >= 2:
            chosen: List[Tuple[str, int]] = []
            for rk in full_racks[:2]:
                for sl in range(rack_size):
                    chosen.append((rk, sl))
            return chosen
        return None

    if nodes_needed > rack_size or (nodes_needed & (nodes_needed - 1)) != 0:
        raise ValueError(f"Buddy sub-ring size must be a power of two (1, 2, 4, 8, 16), got {nodes_needed}")

    candidates: List[Tuple[int, int, str, int]] = []
    parent_size = min(rack_size, nodes_needed * 2)

    for rk in racks_sorted:
        mask = free_mask_by_rack[rk]
        total_free_in_rack = sum(1 for x in mask if x)
        for start in range(0, rack_size, nodes_needed):
            if all(mask[start + off] for off in range(nodes_needed)):
                # Check sibling occupancy inside enclosing buddy parent:
                # 0 free outside block in parent means tightest buddy pair (score 0 = best!)
                parent_start = (start // parent_size) * parent_size
                parent_free = sum(1 for i in range(parent_start, parent_start + parent_size) if mask[i])
                sibling_free = parent_free - nodes_needed
                candidates.append((sibling_free, total_free_in_rack, rk, start))

    if not candidates:
        return None

    candidates.sort()
    _, _, best_rack, best_start = candidates[0]
    return [(best_rack, best_start + off) for off in range(nodes_needed)]


def _find_naive_least_allocated(
    free_mask_by_rack: Dict[str, List[bool]],
    nodes_needed: int,
    allow_cross_rack_spill: bool = True,
) -> Tuple[Optional[List[Tuple[str, int]]], bool]:
    """Baseline Kubernetes-style LeastAllocated placement (picks emptiest racks first, fragmenting rings)."""
    racks_by_emptiest = sorted(
        free_mask_by_rack.keys(),
        key=lambda rk: (-sum(1 for x in free_mask_by_rack[rk] if x), rk),
    )
    total_free = sum(sum(1 for x in m if x) for m in free_mask_by_rack.values())
    if total_free < nodes_needed:
        return None, False

    if not allow_cross_rack_spill and nodes_needed <= 8:
        for rk in racks_by_emptiest:
            free_slots = [i for i, ok in enumerate(free_mask_by_rack[rk]) if ok]
            if len(free_slots) >= nodes_needed:
                picked = [(rk, s) for s in free_slots[:nodes_needed]]
                # Check if picked forms an aligned contiguous sub-ring
                starts = [s for _, s in picked]
                is_contiguous = (
                    max(starts) - min(starts) + 1 == nodes_needed
                    and (min(starts) % nodes_needed == 0)
                )
                return picked, (not is_contiguous)
        return None, False

    # Scatter/pack starting from emptiest rack(s)
    chosen: List[Tuple[str, int]] = []
    racks_touched: Set[str] = set()
    for rk in racks_by_emptiest:
        for slot_idx, ok in enumerate(free_mask_by_rack[rk]):
            if ok:
                chosen.append((rk, slot_idx))
                racks_touched.add(rk)
                if len(chosen) == nodes_needed:
                    break
        if len(chosen) == nodes_needed:
            break

    if len(chosen) < nodes_needed:
        return None, False

    expected_racks = 1 if nodes_needed <= 8 else 2
    fragmented = len(racks_touched) > expected_racks
    if not fragmented and nodes_needed <= 8:
        slots = [s for _, s in chosen]
        if (max(slots) - min(slots) + 1 != nodes_needed) or (min(slots) % nodes_needed != 0):
            fragmented = True
    return chosen, fragmented


def _simulate_policy(
    jobs: Sequence[Dict[str, Any]],
    healthy_mask_init: Dict[str, List[bool]],
    mode: str,
) -> Dict[str, Any]:
    """Run discrete-event gang scheduling simulation for either 'buddy_easy' or 'naive_fifo'."""
    rack_ids = sorted(healthy_mask_init.keys())
    rack_size = len(healthy_mask_init[rack_ids[0]])
    total_healthy_nodes = sum(sum(1 for x in m if x) for m in healthy_mask_init.values())

    # Validate feasibility against total healthy capacity
    pending = [dict(j, arrival=0) for j in jobs]
    running: List[Dict[str, Any]] = []
    completed: List[Dict[str, Any]] = []

    free_mask = {rk: list(healthy_mask_init[rk]) for rk in rack_ids}
    now = 0
    backfilled_count = 0
    fragmented_count = 0
    first_wave_snapshot: Optional[Dict[str, List[str]]] = None

    max_guard_iters = 5000
    iters = 0

    while (pending or running) and iters < max_guard_iters:
        iters += 1
        # Release finished jobs at current timestamp
        still_running: List[Dict[str, Any]] = []
        for rj in running:
            if rj["end_step"] <= now:
                for rk, sl in rj["placement"]:
                    free_mask[rk][sl] = True
                completed.append(rj)
            else:
                still_running.append(rj)
        running = still_running

        if mode == "naive_fifo":
            # Strict FIFO without backfill; LeastAllocated placement (+25% step penalty on fragmented rings!)
            while pending:
                head = pending[0]
                alloc, is_frag = _find_naive_least_allocated(free_mask, int(head["nodes"]), allow_cross_rack_spill=True)
                if alloc is None:
                    break
                pending.pop(0)
                for rk, sl in alloc:
                    free_mask[rk][sl] = False
                eff_dur = int(head["duration"]) + (
                    max(1, math.ceil(int(head["duration"]) * 0.25)) if is_frag and int(head["nodes"]) > 1 else 0
                )
                if is_frag and int(head["nodes"]) > 1:
                    fragmented_count += 1
                running.append(
                    {
                        **head,
                        "start_step": now,
                        "end_step": now + eff_dur,
                        "wait_steps": now - int(head["arrival"]),
                        "placement": alloc,
                        "fragmented": is_frag and int(head["nodes"]) > 1,
                        "backfilled": False,
                    }
                )
        else:
            # Buddy Sub-Ring + EASY Backfill
            made_progress = True
            while pending and made_progress:
                made_progress = False
                head = pending[0]
                alloc = _find_buddy_allocation(free_mask, int(head["nodes"]), rack_size=rack_size)
                if alloc is not None:
                    pending.pop(0)
                    for rk, sl in alloc:
                        free_mask[rk][sl] = False
                    running.append(
                        {
                            **head,
                            "start_step": now,
                            "end_step": now + int(head["duration"]),
                            "wait_steps": now - int(head["arrival"]),
                            "placement": alloc,
                            "fragmented": False,
                            "backfilled": False,
                        }
                    )
                    made_progress = True
                else:
                    # Compute Shadow Start Time T_shadow and shadow reservation for head-of-line job
                    future_Mask = {rk: list(free_mask[rk]) for rk in rack_ids}
                    shadow_time: Optional[int] = None
                    shadow_alloc: Optional[List[Tuple[str, int]]] = None
                    for rj in sorted(running, key=lambda x: x["end_step"]):
                        for rk, sl in rj["placement"]:
                            future_Mask[rk][sl] = True
                        cand_alloc = _find_buddy_allocation(
                            future_Mask, int(head["nodes"]), rack_size=rack_size
                        )
                        if cand_alloc is not None:
                            shadow_time = rj["end_step"]
                            shadow_alloc = cand_alloc
                            break

                    if shadow_time is None or shadow_alloc is None:
                        # Head job cannot fit even when all running jobs drain (e.g. too many cordoned nodes)
                        break

                    shadow_set = set(shadow_alloc)
                    # Scan queue behind head for EASY backfill candidates
                    idx = 1
                    while idx < len(pending):
                        cand = pending[idx]
                        c_dur = int(cand["duration"])
                        c_nodes = int(cand["nodes"])
                        if now + c_dur <= shadow_time:
                            # Finishes before head job's shadow start -> can use ANY currently free buddy sub-ring
                            bf_alloc = _find_buddy_allocation(free_mask, c_nodes, rack_size=rack_size)
                        else:
                            # Runs past shadow_time -> can ONLY use free slots NOT in shadow_set
                            non_reserved_mask = {
                                rk: [
                                    free_mask[rk][s] and ((rk, s) not in shadow_set)
                                    for s in range(rack_size)
                                ]
                                for rk in rack_ids
                            }
                            bf_alloc = _find_buddy_allocation(non_reserved_mask, c_nodes, rack_size=rack_size)

                        if bf_alloc is not None:
                            pending.pop(idx)
                            for rk, sl in bf_alloc:
                                free_mask[rk][sl] = False
                            backfilled_count += 1
                            running.append(
                                {
                                    **cand,
                                    "start_step": now,
                                    "end_step": now + c_dur,
                                    "wait_steps": now - int(cand["arrival"]),
                                    "placement": bf_alloc,
                                    "fragmented": False,
                                    "backfilled": True,
                                }
                            )
                        else:
                            idx += 1
                    break

        if first_wave_snapshot is None and running:
            snap: Dict[str, List[str]] = {
                rk: [
                    ("CORDONED" if not healthy_mask_init[rk][s] else "IDLE")
                    for s in range(rack_size)
                ]
                for rk in rack_ids
            }
            for rj in running:
                for rk, sl in rj["placement"]:
                    snap[rk][sl] = rj["job_id"]
            first_wave_snapshot = snap

        if not running:
            if pending:
                # Unschedulable head job due to cordoned rack capacity limits
                unsched = pending.pop(0)
                completed.append(
                    {
                        **unsched,
                        "start_step": now,
                        "end_step": now,
                        "wait_steps": 0,
                        "placement": [],
                        "fragmented": False,
                        "backfilled": False,
                        "unschedulable": True,
                    }
                )
            continue

        now = min(rj["end_step"] for rj in running)

    makespan = max((c["end_step"] for c in completed), default=1)
    useful_node_steps = sum(
        int(c["nodes"]) * int(c["duration"]) for c in completed if not c.get("unschedulable")
    )
    available_node_steps = max(1, total_healthy_nodes * max(1, makespan))
    utilization_pct = (useful_node_steps / available_node_steps) * 100.0
    waits = sorted(int(c["wait_steps"]) for c in completed if not c.get("unschedulable"))
    p50_wait = waits[len(waits) // 2] if waits else 0
    p95_wait = waits[min(len(waits) - 1, int(len(waits) * 0.95))] if waits else 0

    return {
        "mode": mode,
        "makespan_steps": makespan,
        "utilization_pct": round(utilization_pct, 2),
        "p50_wait_steps": p50_wait,
        "p95_wait_steps": p95_wait,
        "mean_wait_steps": round(statistics.mean(waits), 2) if waits else 0.0,
        "cross_spine_fragmented_jobs": fragmented_count,
        "backfilled_jobs": backfilled_count,
        "initial_rack_map": first_wave_snapshot or {},
        "schedule": [
            {
                "job_id": c["job_id"],
                "kind": c.get("kind", "Workload"),
                "nodes": c["nodes"],
                "duration": c["duration"],
                "start_step": c["start_step"],
                "end_step": c["end_step"],
                "wait_steps": c["wait_steps"],
                "backfilled": c["backfilled"],
                "fragmented": c["fragmented"],
                "racks": sorted({rk for rk, _ in c["placement"]}),
                "slots": [f"{rk}:{sl}" for rk, sl in c["placement"]],
            }
            for c in sorted(completed, key=lambda x: (x["start_step"], x["job_id"]))
        ],
    }


def schedule_workload_queue(
    preset_name: str = "mixed_training_and_ci",
    jobs: Optional[Sequence[Dict[str, Any]]] = None,
    cordoned_hosts: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compare Buddy Sub-Ring + EASY Backfill against Naive LeastAllocated + Strict FIFO."""
    t0 = time.perf_counter()
    rack_ids = ["rack-01", "rack-02", "rack-03", "rack-04"]
    healthy_mask: Dict[str, List[bool]] = {rk: [True] * 8 for rk in rack_ids}

    cordoned_list = list(cordoned_hosts) if cordoned_hosts is not None else ["usv-r01-h03", "usv-r02-h05"]
    for hid in cordoned_list:
        # Format usv-r01-h03 -> rack-01, slot 3
        if hid.startswith("usv-r0") and "-h0" in hid:
            try:
                r_part, h_part = hid.replace("usv-r0", "").split("-h0")
                rk = f"rack-0{int(r_part)}"
                sl = int(h_part)
                if rk in healthy_mask and 0 <= sl < 8:
                    healthy_mask[rk][sl] = False
            except ValueError:
                pass

    workload = list(jobs) if jobs is not None else DEFAULT_WORKLOAD_PRESETS.get(
        preset_name, DEFAULT_WORKLOAD_PRESETS["mixed_training_and_ci"]
    )

    buddy_res = _simulate_policy(workload, healthy_mask, mode="buddy_easy")
    naive_res = _simulate_policy(workload, healthy_mask, mode="naive_fifo")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    util_gain_pts = round(buddy_res["utilization_pct"] - naive_res["utilization_pct"], 2)
    wait_reduction_pct = (
        round((1.0 - buddy_res["mean_wait_steps"] / naive_res["mean_wait_steps"]) * 100.0, 2)
        if naive_res["mean_wait_steps"] > 0
        else 0.0
    )

    return {
        "preset_name": preset_name,
        "total_cluster_nodes": 32,
        "cordoned_hosts": cordoned_list,
        "healthy_available_nodes": 32 - len(cordoned_list),
        "jobs_count": len(workload),
        "solve_ms": round(elapsed_ms, 3),
        "comparison": {
            "buddy_easy_utilization_pct": buddy_res["utilization_pct"],
            "naive_fifo_utilization_pct": naive_res["utilization_pct"],
            "utilization_gain_pct_pts": util_gain_pts,
            "buddy_easy_makespan_steps": buddy_res["makespan_steps"],
            "naive_fifo_makespan_steps": naive_res["makespan_steps"],
            "buddy_easy_mean_wait_steps": buddy_res["mean_wait_steps"],
            "naive_fifo_mean_wait_steps": naive_res["mean_wait_steps"],
            "mean_wait_reduction_pct": wait_reduction_pct,
            "buddy_cross_spine_fragmentations": buddy_res["cross_spine_fragmented_jobs"],
            "naive_cross_spine_fragmentations": naive_res["cross_spine_fragmented_jobs"],
            "easy_backfilled_jobs": buddy_res["backfilled_jobs"],
        },
        "buddy_easy": buddy_res,
        "naive_fifo": naive_res,
    }


def run_scheduler_monte_carlo_benchmark(
    num_workloads: int = 40,
    jobs_per_workload: int = 18,
    seed: int = 20261125,
) -> Dict[str, Any]:
    """Benchmark Buddy Sub-Ring + EASY Backfill vs. Naive FIFO across 40 mixed cluster queues."""
    rng = random.Random(seed)
    buddy_utils: List[float] = []
    naive_utils: List[float] = []
    buddy_waits: List[float] = []
    naive_waits: List[float] = []
    buddy_frags = 0
    naive_frags = 0
    total_backfilled = 0
    solve_times: List[float] = []

    all_hosts = [f"usv-r0{r}-h0{s}" for r in range(1, 5) for s in range(8)]

    for w_idx in range(num_workloads):
        # Cordon 2 random hosts across distinct racks to simulate active maintenance
        cordon = [
            f"usv-r01-h0{rng.randrange(8)}",
            f"usv-r03-h0{rng.randrange(8)}",
        ]
        jobs = []
        for j_idx in range(jobs_per_workload):
            # Realistic mixed distribution: large 8/16-node gangs + short 1/2/4-node validation/LoRA jobs
            nodes = rng.choices([1, 2, 4, 8, 16], weights=[28, 26, 20, 18, 8], k=1)[0]
            dur = rng.randint(2, 5) if nodes <= 2 else rng.randint(6, 12)
            jobs.append(
                {
                    "job_id": f"w{w_idx:02d}-j{j_idx:02d}",
                    "nodes": nodes,
                    "duration": dur,
                    "kind": f"{nodes}-Node Gang",
                }
            )
        out = schedule_workload_queue(preset_name=f"mc-{w_idx}", jobs=jobs, cordoned_hosts=cordon)
        solve_times.append(out["solve_ms"])
        cmp_m = out["comparison"]
        buddy_utils.append(cmp_m["buddy_easy_utilization_pct"])
        naive_utils.append(cmp_m["naive_fifo_utilization_pct"])
        buddy_waits.append(cmp_m["buddy_easy_mean_wait_steps"])
        naive_waits.append(cmp_m["naive_fifo_mean_wait_steps"])
        buddy_frags += cmp_m["buddy_cross_spine_fragmentations"]
        naive_frags += cmp_m["naive_cross_spine_fragmentations"]
        total_backfilled += cmp_m["easy_backfilled_jobs"]

    mean_buddy_u = statistics.mean(buddy_utils)
    mean_naive_u = statistics.mean(naive_utils)
    mean_buddy_w = statistics.mean(buddy_waits)
    mean_naive_w = statistics.mean(naive_waits)
    st_sorted = sorted(solve_times)

    return {
        "num_workloads": num_workloads,
        "jobs_per_workload": jobs_per_workload,
        "total_jobs_scheduled": num_workloads * jobs_per_workload,
        "buddy_easy_mean_utilization_pct": round(mean_buddy_u, 2),
        "naive_fifo_mean_utilization_pct": round(mean_naive_u, 2),
        "mean_utilization_gain_pct_pts": round(mean_buddy_u - mean_naive_u, 2),
        "buddy_easy_mean_wait_steps": round(mean_buddy_w, 2),
        "naive_fifo_mean_wait_steps": round(mean_naive_w, 2),
        "mean_wait_reduction_pct": round((1.0 - mean_buddy_w / max(0.01, mean_naive_w)) * 100.0, 2),
        "buddy_cross_spine_fragmentations": buddy_frags,
        "naive_cross_spine_fragmentations": naive_frags,
        "total_easy_backfilled_jobs": total_backfilled,
        "solve_p50_ms": round(st_sorted[len(st_sorted) // 2], 3),
        "solve_p95_ms": round(st_sorted[min(len(st_sorted) - 1, int(len(st_sorted) * 0.95))], 3),
    }
