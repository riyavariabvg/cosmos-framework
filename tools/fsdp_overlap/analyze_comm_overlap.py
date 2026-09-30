#!/usr/bin/env python3
"""Computation–communication overlap analysis for torch.profiler traces of FSDP2 training.

Two independent measurements, so they can cross-check each other:

1. HolisticTraceAnalysis (if installed: `pip install HolisticTraceAnalysis`):
   get_comm_comp_overlap() and get_temporal_breakdown().
2. A stdlib-only analysis of the raw Chrome traces, which additionally reports
   WHERE communication is exposed (collective type, forward/backward/optimizer,
   the FSDP2 record_function label incl. module FQN, position within the step)
   and estimates how much NCCL kernel time is really waiting for other ranks.

Definitions (per rank, over the captured GPU window):
  compute   = union of all non-NCCL GPU kernels (any stream)
  comm      = union of NCCL kernels
  overlap % = |comm ∩ compute| / |comm|            (same notion as HTA)
  exposed   = |comm \\ compute|  -> GPU is doing communication and nothing else
  idle      = window - |comm ∪ compute|          -> GPU doing nothing at all

Caveat: an NCCL kernel's duration includes spin-waiting for the slowest rank.
"est. wait" below = kernel dur minus the shortest duration of the same
collective across ranks; large values mean rank skew / load imbalance, not bandwidth.

Usage:
  python analyze_comm_overlap.py --trace-dir <dir with rank*_trace.json.gz> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import gzip
import json
import os
import re
import sys
from pathlib import Path

COLL_RE = re.compile(r"(AllGather|ReduceScatter|AllReduce|Broadcast|Reduce|SendRecv|AllToAll|Send|Recv)", re.I)
CPU_CATS = {"cpu_op", "user_annotation", "python_function"}
LAUNCH_NAMES = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx", "cudaLaunchCooperativeKernel")


# ------------------------------------------------------------------ interval helpers
def merge(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


def total(iv):
    return sum(e - s for s, e in iv)


def intersect_len(a, b):
    """Length of intersection of two merged interval lists."""
    i = j = 0
    t = 0.0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            t += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return t


def covered_len(s, e, merged, starts):
    """Length of [s,e] covered by a merged interval list."""
    k = max(bisect.bisect_right(starts, s) - 1, 0)
    t = 0.0
    while k < len(merged) and merged[k][0] < e:
        a, b = max(s, merged[k][0]), min(e, merged[k][1])
        if b > a:
            t += b - a
        k += 1
    return t


def clip(iv, lo, hi):
    return [[max(s, lo), min(e, hi)] for s, e in iv if e > lo and s < hi]


# ------------------------------------------------------------------ trace loading
def load_trace(path: Path):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        return json.load(f)


def rank_of(trace, path: Path):
    di = trace.get("distributedInfo") or {}
    if "rank" in di:
        return int(di["rank"])
    m = re.search(r"rank(\d+)", path.name)
    return int(m.group(1)) if m else -1


def is_nccl(name: str) -> bool:
    n = name.lower()
    return "nccl" in n


def coll_type(name: str) -> str:
    m = COLL_RE.search(name)
    return m.group(1) if m else "Other"


def short_label(stack):
    """Condense an enclosing CPU-op stack into (phase, fsdp_label, fsdp_label_with_fqn)."""
    phase = "forward"
    fsdp = [n for n in stack if n.startswith("FSDP::")]
    for n in stack:
        if n.startswith("autograd::engine") or "backward" in n.lower():
            phase = "backward"
        if "Optimizer.step" in n or ("optimizer" in n.lower() and "step" in n.lower()):
            phase = "optimizer"
            break
    full = fsdp[-1] if fsdp else (stack[-1] if stack else "?")
    generic = re.sub(r"\s*\(.*\)\s*$", "", full)
    return phase, generic, full


def analyze_rank(trace, skip_steps: int):
    ev = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X" and "dur" in e]
    kernels = [e for e in ev if e.get("cat") == "kernel"]
    if not kernels:
        raise RuntimeError("no GPU kernel events in trace (was CUDA activity enabled?)")

    # CPU launch time of each kernel, via correlation id
    launch = {}
    for e in ev:
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and e.get("name", "").startswith(LAUNCH_NAMES):
            c = e.get("args", {}).get("correlation")
            if c is not None:
                launch[c] = (e["ts"], e.get("tid"), e.get("pid"))

    # Profiler steps (CPU side) -> assign kernels to steps by launch time
    steps = sorted(
        (e for e in ev if e.get("name", "").startswith("ProfilerStep#")), key=lambda e: e["ts"]
    )
    step_bounds = [(int(e["name"].split("#")[1]), e["ts"], e["ts"] + e["dur"]) for e in steps]

    def step_of(ts):
        for sid, s, e in step_bounds:
            if s <= ts < e:
                return sid
        return None

    # Enclosing CPU op stacks for NCCL launches (sweep per thread)
    nccl_k = [k for k in kernels if is_nccl(k.get("name", ""))]
    queries = collections.defaultdict(list)  # (pid,tid) -> [(ts, corr)]
    for k in nccl_k:
        c = k.get("args", {}).get("correlation")
        if c in launch:
            ts, tid, pid = launch[c]
            queries[(pid, tid)].append((ts, c))
    cpu_by_thread = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") in CPU_CATS and (e.get("pid"), e.get("tid")) in queries:
            cpu_by_thread[(e["pid"], e["tid"])].append((e["ts"], e["ts"] + e["dur"], e["name"]))
    stacks = {}
    for key, qs in queries.items():
        cpu = sorted(cpu_by_thread.get(key, []), key=lambda x: (x[0], -x[1]))
        qs.sort()
        stack, i = [], 0
        for ts, c in qs:
            while i < len(cpu) and cpu[i][0] <= ts:
                while stack and stack[-1][1] <= cpu[i][0]:
                    stack.pop()
                stack.append(cpu[i])
                i += 1
            while stack and stack[-1][1] <= ts:
                stack.pop()
            stacks[c] = [s[2] for s in stack if s[0] <= ts < s[1]]

    # Tag kernels with step; drop skipped steps
    valid_steps = sorted({sid for sid, _, _ in step_bounds})[skip_steps:]
    valid = set(valid_steps)

    def kstep(k):
        c = k.get("args", {}).get("correlation")
        return step_of(launch[c][0]) if c in launch else None

    per_step = collections.defaultdict(lambda: {"comp": [], "comm": [], "nccl": []})
    for k in kernels:
        sid = kstep(k)
        if step_bounds and sid not in valid:
            continue
        sid = sid if step_bounds else -1
        iv = [k["ts"], k["ts"] + k["dur"]]
        if is_nccl(k["name"]):
            per_step[sid]["comm"].append(iv)
            per_step[sid]["nccl"].append(k)
        else:
            per_step[sid]["comp"].append(iv)

    t0 = min(e["ts"] for e in ev)
    step_rows, exposed_rows = [], []
    agg = collections.Counter()
    tot = collections.Counter()
    for sid in sorted(per_step, key=lambda x: (x is None, x)):
        d = per_step[sid]
        comp, comm = merge(d["comp"]), merge(d["comm"])
        if not comp and not comm:
            continue
        lo = min([iv[0] for iv in comp + comm])
        hi = max([iv[1] for iv in comp + comm])
        window = hi - lo
        c_comm, c_comp = total(comm), total(comp)
        ov = intersect_len(comm, comp)
        busy = total(merge(comp + comm))
        row = dict(
            step=sid, window_ms=window / 1e3, compute_ms=c_comp / 1e3, comm_ms=c_comm / 1e3,
            overlap_ms=ov / 1e3, exposed_comm_ms=(c_comm - ov) / 1e3, idle_ms=(window - busy) / 1e3,
            overlap_pct=100 * ov / c_comm if c_comm else float("nan"),
            exposed_pct_of_step=100 * (c_comm - ov) / window if window else 0.0,
        )
        step_rows.append(row)
        for kk in ("window_ms", "compute_ms", "comm_ms", "overlap_ms", "exposed_comm_ms", "idle_ms"):
            tot[kk] += row[kk]

        starts = [s for s, _ in comp]
        for k in d["nccl"]:
            s, e = k["ts"], k["ts"] + k["dur"]
            exp = (e - s) - covered_len(s, e, comp, starts)
            c = k.get("args", {}).get("correlation")
            phase, generic, full = short_label(stacks.get(c, []))
            ctype = coll_type(k["name"])
            agg[(ctype, phase, generic, "exposed")] += exp
            agg[(ctype, phase, generic, "total")] += e - s
            agg[(ctype, phase, generic, "count")] += 1
            exposed_rows.append(dict(
                step=sid, collective=ctype, phase=phase, label=full, stream=k.get("tid"),
                dur_us=round(e - s, 1), exposed_us=round(exp, 1),
                offset_ms_from_trace_start=round((s - t0) / 1e3, 3),
                pos_in_step_pct=round(100 * (s - lo) / window, 1) if window else 0.0,
                corr=c, kernel=k["name"][:120],
            ))
    return step_rows, exposed_rows, agg, tot


def est_wait(all_exposed):
    """Match NCCL kernels across ranks by launch order and estimate skew-wait."""
    seqs = {r: sorted(rows, key=lambda x: (x["corr"] is None, x["corr"] or 0)) for r, rows in all_exposed.items()}
    lens = {len(v) for v in seqs.values()}
    if len(seqs) < 2 or len(lens) != 1:
        return None
    ranks = sorted(seqs)
    wait = collections.Counter()
    for i in range(lens.pop()):
        items = [seqs[r][i] for r in ranks]
        if len({it["collective"] for it in items}) != 1:
            return None  # sequences don't line up; bail rather than report garbage
        m = min(it["dur_us"] for it in items)
        for r, it in zip(ranks, items):
            it["est_wait_us"] = round(it["dur_us"] - m, 1)
            wait[r] += it["dur_us"] - m
    return wait


def run_hta(trace_dir: Path, out: Path, lines):
    try:
        from hta.trace_analysis import TraceAnalysis
    except ImportError:
        lines.append("\n## HTA\nHolisticTraceAnalysis not installed (`pip install HolisticTraceAnalysis`); skipped.\n")
        return
    lines.append("\n## HTA\n")
    try:
        ta = TraceAnalysis(trace_dir=str(trace_dir))
        ov = ta.get_comm_comp_overlap(visualize=False)
        ov.to_csv(out / "hta_comm_comp_overlap.csv", index=False)
        lines.append("get_comm_comp_overlap() (% of comm time overlapped with compute):\n\n```\n" + ov.to_string(index=False) + "\n```\n")
        tb = ta.get_temporal_breakdown(visualize=False)
        tb.to_csv(out / "hta_temporal_breakdown.csv", index=False)
        lines.append("get_temporal_breakdown():\n\n```\n" + tb.to_string(index=False) + "\n```\n")
    except Exception as ex:  # HTA is brittle across torch trace versions
        lines.append(f"HTA failed: `{type(ex).__name__}: {ex}` — rely on the manual analysis below.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--skip-steps", type=int, default=0, help="extra captured steps to drop from the front")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in a.trace_dir.iterdir() if re.match(r".*trace.*\.json(\.gz)?$", p.name))
    if not files:
        sys.exit(f"no trace files in {a.trace_dir}")

    lines = [f"# FSDP2 comm–comp overlap\n\nTraces: `{a.trace_dir}` ({len(files)} ranks)\n"]
    run_hta(a.trace_dir, a.out_dir, lines)

    all_steps, all_exposed, all_agg, all_tot = {}, {}, {}, {}
    for p in files:
        tr = load_trace(p)
        r = rank_of(tr, p)
        s, e, g, t = analyze_rank(tr, a.skip_steps)
        all_steps[r], all_exposed[r], all_agg[r], all_tot[r] = s, e, g, t
        del tr
    wait = est_wait(all_exposed)

    lines.append("\n## Manual analysis — per rank (summed over analyzed steps)\n")
    lines.append("| rank | window ms | compute ms | comm ms | overlap % | exposed comm ms | exposed % of window | idle ms | est. NCCL wait ms |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    summary = {}
    for r in sorted(all_tot):
        t = all_tot[r]
        ovp = 100 * t["overlap_ms"] / t["comm_ms"] if t["comm_ms"] else float("nan")
        exw = 100 * t["exposed_comm_ms"] / t["window_ms"] if t["window_ms"] else 0
        w = wait[r] / 1e3 if wait else float("nan")
        summary[r] = dict(t, overlap_pct=ovp, exposed_pct_of_window=exw, est_wait_ms=w)
        lines.append(f"| {r} | {t['window_ms']:.1f} | {t['compute_ms']:.1f} | {t['comm_ms']:.1f} | {ovp:.1f} | "
                     f"{t['exposed_comm_ms']:.1f} | {exw:.1f} | {t['idle_ms']:.1f} | {w:.1f} |")
    if wait is None:
        lines.append("\n(est. NCCL wait unavailable: NCCL kernel sequences don't line up across ranks.)")

    lines.append("\n## Where exposed communication happens (rank 0; summed over steps)\n")
    r0 = min(all_agg)
    g = all_agg[r0]
    keys = sorted({k[:3] for k in g}, key=lambda k: -g[k + ("exposed",)])
    lines.append("| collective | phase | FSDP label | count | total ms | exposed ms | exposed % |")
    lines.append("|---|---|---|---|---|---|---|")
    for k in keys:
        tt, ex, n = g[k + ("total",)], g[k + ("exposed",)], g[k + ("count",)]
        lines.append(f"| {k[0]} | {k[1]} | `{k[2]}` | {n} | {tt/1e3:.2f} | {ex/1e3:.2f} | {100*ex/tt if tt else 0:.0f} |")

    lines.append(f"\n## Top {a.top} exposed NCCL kernels, rank {r0} (jump to these in Perfetto)\n")
    lines.append("| offset ms | step | pos in step % | collective | phase | label | dur µs | exposed µs | est. wait µs |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for x in sorted(all_exposed[r0], key=lambda x: -x["exposed_us"])[: a.top]:
        lines.append(f"| {x['offset_ms_from_trace_start']} | {x['step']} | {x['pos_in_step_pct']} | {x['collective']} | "
                     f"{x['phase']} | `{x['label'][:70]}` | {x['dur_us']} | {x['exposed_us']} | {x.get('est_wait_us', '')} |")

    # machine-readable outputs
    rows = [{"rank": r, **row} for r in sorted(all_steps) for row in all_steps[r]]
    if rows:
        with open(a.out_dir / "per_step.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(a.out_dir / "nccl_kernels.csv", "w", newline="") as f:
        fields = ["rank", "step", "collective", "phase", "label", "stream", "dur_us", "exposed_us", "est_wait_us",
                  "offset_ms_from_trace_start", "pos_in_step_pct", "corr", "kernel"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        for r in sorted(all_exposed):
            for x in all_exposed[r]:
                w.writerow({"rank": r, **x})
    (a.out_dir / "summary.json").write_text(json.dumps({str(k): v for k, v in summary.items()}, indent=2))
    (a.out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {a.out_dir}/summary.md, per_step.csv, nccl_kernels.csv, summary.json")


if __name__ == "__main__":
    main()
