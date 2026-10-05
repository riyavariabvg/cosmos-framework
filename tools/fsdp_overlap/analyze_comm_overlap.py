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

Skew vs. residual (added on top of the metrics above, which are unchanged):
  An NCCL kernel spins from launch until the last rank arrives, then transfers.
  So the first `est. wait` µs of each kernel are treated as skew-wait, the rest as transfer.
  exposed = skew-wait exposed (exposed time inside those first est. wait µs)
          + residual exposed  (everything else; an upper bound on exposed transfer,
                               since the fastest rank's duration can still contain waiting)
  For the top exposed kernels it also reports what the waited-for rank (the last to
  launch that collective) was running in the same wall-clock window. This compares
  timestamps across ranks, which is only meaningful when all ranks share a host clock
  (single node); across nodes the window can be shifted by clock offset.

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
import re
import sys
from pathlib import Path

COLL_RE = re.compile(r"(AllGather|ReduceScatter|AllReduce|Broadcast|Reduce|SendRecv|AllToAll|Send|Recv)", re.I)
CPU_CATS = {"cpu_op", "user_annotation", "python_function"}
LAUNCH_NAMES = (
    "cudaLaunchKernel",
    "cudaLaunchKernelExC",
    "cuLaunchKernel",
    "cuLaunchKernelEx",
    "cudaLaunchCooperativeKernel",
)


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


def subtract(a, b):
    """a \\ b for two merged interval lists."""
    out, j = [], 0
    for s, e in a:
        cur = s
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            if b[k][0] > cur:
                out.append([cur, b[k][0]])
            cur = max(cur, b[k][1])
            k += 1
        if cur < e:
            out.append([cur, e])
    return out


def kernel_label(name: str, in_conv3d: bool = False) -> str:
    """Short, groupable name for a GPU kernel."""
    if is_nccl(name):
        return f"NCCL {coll_type(name)}"
    if in_conv3d:
        return "conv3d (VAE encode)"
    n = re.sub(r"^void\s+", "", name)
    n = re.sub(r"<.*", "", n)
    n = re.sub(r"\(.*", "", n)
    return n[:60]


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
    steps = sorted((e for e in ev if e.get("name", "").startswith("ProfilerStep#")), key=lambda e: e["ts"])
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
    step_iv = {}  # sid -> (merged compute, its starts, exposed comm intervals)
    for sid in sorted(per_step, key=lambda x: (x is None, x)):
        d = per_step[sid]
        comp, comm = merge(d["comp"]), merge(d["comm"])
        if not comp and not comm:
            continue
        step_iv[sid] = (comp, [s for s, _ in comp], subtract(comm, comp))
        lo = min([iv[0] for iv in comp + comm])
        hi = max([iv[1] for iv in comp + comm])
        window = hi - lo
        c_comm, c_comp = total(comm), total(comp)
        ov = intersect_len(comm, comp)
        busy = total(merge(comp + comm))
        row = dict(
            step=sid,
            window_ms=window / 1e3,
            compute_ms=c_comp / 1e3,
            comm_ms=c_comm / 1e3,
            overlap_ms=ov / 1e3,
            exposed_comm_ms=(c_comm - ov) / 1e3,
            idle_ms=(window - busy) / 1e3,
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
            exposed_rows.append(
                dict(
                    step=sid,
                    collective=ctype,
                    phase=phase,
                    label=full,
                    stream=k.get("tid"),
                    dur_us=round(e - s, 1),
                    exposed_us=round(exp, 1),
                    offset_ms_from_trace_start=round((s - t0) / 1e3, 3),
                    pos_in_step_pct=round(100 * (s - lo) / window, 1) if window else 0.0,
                    corr=c,
                    kernel=k["name"][:120],
                    ts_us=s,
                )
            )

    # Context for cross-rank "what was the other rank doing" queries.
    conv_ranges = collections.defaultdict(list)  # (pid,tid) -> sorted aten::conv3d CPU ranges
    for e in ev:
        if e.get("cat") == "cpu_op" and e["name"] == "aten::conv3d":
            conv_ranges[(e["pid"], e["tid"])].append((e["ts"], e["ts"] + e["dur"]))
    conv_starts = {}
    for key in conv_ranges:
        conv_ranges[key].sort()
        conv_starts[key] = [s for s, _ in conv_ranges[key]]

    def launched_in_conv3d(k):
        c = k.get("args", {}).get("correlation")
        if c not in launch:
            return False
        ts, tid, pid = launch[c]
        rs = conv_ranges.get((pid, tid))
        if not rs:
            return False
        i = bisect.bisect_right(conv_starts[(pid, tid)], ts) - 1
        return i >= 0 and rs[i][0] <= ts < rs[i][1]

    gpu = sorted(
        (k["ts"], k["ts"] + k["dur"], kernel_label(k["name"], launched_in_conv3d(k)), is_nccl(k["name"]))
        for k in kernels
    )
    cpu = sorted(
        (e["ts"], e["ts"] + e["dur"], e["name"])
        for e in ev
        if e.get("cat") in ("cpu_op", "user_annotation") and not e["name"].startswith("ProfilerStep#")
    )
    ctx = dict(
        t0=t0,
        step_iv=step_iv,
        gpu=gpu,
        gpu_starts=[g[0] for g in gpu],
        gpu_maxdur=max((g[1] - g[0] for g in gpu), default=0),
        cpu=cpu,
        cpu_starts=[c[0] for c in cpu],
        cpu_maxdur=max((c[1] - c[0] for c in cpu), default=0),
    )
    return step_rows, exposed_rows, agg, tot, ctx


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
        last = max(ranks, key=lambda r: seqs[r][i]["ts_us"])  # last rank to launch = the one waited for
        for r, it in zip(ranks, items):
            it["est_wait_us"] = round(it["dur_us"] - m, 1)
            it["match_idx"] = i
            it["waited_for_rank"] = (
                last if last != r else max((o for o in ranks if o != r), key=lambda o: seqs[o][i]["ts_us"])
            )
            wait[r] += it["dur_us"] - m
    return wait


def split_skew(all_exposed, all_ctx):
    """Split exposed comm into skew-wait (first est. wait µs of each NCCL kernel) and residual.

    Per kernel and per rank/step; per-rank totals use the same union-based exposed time as the
    "exposed comm ms" metric, so skew + residual == exposed exactly.
    """
    per_step = collections.defaultdict(dict)
    for r, rows in all_exposed.items():
        by_step = collections.defaultdict(list)
        for x in rows:
            by_step[x["step"]].append(x)
        for sid, xs in by_step.items():
            comp, starts, exposed_iv = all_ctx[r]["step_iv"][sid]
            wait_iv = []
            for x in xs:
                w = min(x.get("est_wait_us", 0.0), x["dur_us"])
                s = x["ts_us"]
                x["skew_exposed_us"] = round(w - covered_len(s, s + w, comp, starts), 1) if w > 0 else 0.0
                x["residual_exposed_us"] = round(x["exposed_us"] - x["skew_exposed_us"], 1)
                if w > 0:
                    wait_iv.append([s, s + w])
            skew = intersect_len(exposed_iv, merge(wait_iv))
            per_step[r][sid] = (skew / 1e3, (total(exposed_iv) - skew) / 1e3)
    return per_step


def in_window(ctx, kind, lo, hi):
    """(start, end, name, ...) records of ctx[kind] overlapping [lo, hi]."""
    recs, starts, maxdur = ctx[kind], ctx[f"{kind}_starts"], ctx[f"{kind}_maxdur"]
    i = bisect.bisect_left(starts, lo - maxdur)
    j = bisect.bisect_left(starts, hi)
    return [x for x in recs[i:j] if x[1] > lo]


def describe_window(ctx, lo, hi, n=3):
    """What a rank was running in [lo, hi]: GPU compute busy %, top GPU work, top CPU ops."""
    gpu = in_window(ctx, "gpu", lo, hi)
    by_kernel = collections.Counter()
    for s, e, label, _ in gpu:
        by_kernel[label] += min(e, hi) - max(s, lo)
    busy = total(merge([[max(s, lo), min(e, hi)] for s, e, _, nc in gpu if not nc]))
    by_cpu = collections.Counter()
    for s, e, name in in_window(ctx, "cpu", lo, hi):
        by_cpu[name] += min(e, hi) - max(s, lo)

    def fmt(c):
        return "; ".join(f"{k[:55]} {v / 1e3:.1f}ms" for k, v in c.most_common(n))

    return 100 * busy / (hi - lo) if hi > lo else 0.0, fmt(by_kernel), fmt(by_cpu)


def run_hta(trace_dir: Path, out: Path, lines):
    try:
        from hta.trace_analysis import TraceAnalysis
    except ImportError:
        lines.append("\n## HTA\nHolisticTraceAnalysis not installed (`pip install HolisticTraceAnalysis`); skipped.\n")
        return
    lines.append("\n## HTA\n")
    try:
        # HTA drops kernels from the last ProfilerStep by default; keep them so HTA
        # covers the same steps as the manual analysis below.
        ta = TraceAnalysis(trace_dir=str(trace_dir), include_last_profiler_step=True)
        ov = ta.get_comm_comp_overlap(visualize=False)
        ov.to_csv(out / "hta_comm_comp_overlap.csv", index=False)
        lines.append(
            "get_comm_comp_overlap() (% of comm time overlapped with compute):\n\n```\n"
            + ov.to_string(index=False)
            + "\n```\n"
        )
        tb = ta.get_temporal_breakdown(visualize=False)
        tb.to_csv(out / "hta_temporal_breakdown.csv", index=False)
        lines.append("get_temporal_breakdown():\n\n```\n" + tb.to_string(index=False) + "\n```\n")
    except Exception as ex:  # HTA is brittle across torch trace versions
        lines.append(f"HTA failed: `{type(ex).__name__}: {ex}` — rely on the manual analysis below.\n")
        return
    # With trainer.profiling.record_shape=true, Inductor's Triton launch ops (CPU events) carry a
    # `stream` arg, and HTA counts them as GPU compute, inflating overlap. Report HTA again with
    # those rows removed; the unfiltered result above is kept as-is.
    try:
        from hta.analyzers.communication_analysis import CommunicationAnalysis

        sym = ta.t.symbol_table.get_sym_table()
        dropped = 0
        for r, df in list(ta.t.traces.items()):
            bad = df["stream"].ne(-1) & df["cat"].map(lambda i: sym[i] == "cpu_op")
            dropped += int(bad.sum())
            ta.t.traces[r] = df[~bad]
        if dropped:
            ov2 = CommunicationAnalysis.get_comm_comp_overlap(ta.t, visualize=False)
            ov2.to_csv(out / "hta_comm_comp_overlap_gpu_only.csv", index=False)
            tb2 = ta.get_temporal_breakdown(visualize=False)
            tb2.to_csv(out / "hta_temporal_breakdown_gpu_only.csv", index=False)
            lines.append(
                f"With {dropped} stream-tagged CPU-op rows removed (HTA misreads these as GPU compute "
                "in record_shape traces):\n\nget_comm_comp_overlap():\n\n```\n"
                + ov2.to_string(index=False)
                + "\n```\n\nget_temporal_breakdown():\n\n```\n"
                + tb2.to_string(index=False)
                + "\n```\n"
            )
    except Exception as ex:
        lines.append(f"HTA GPU-only re-run failed: `{type(ex).__name__}: {ex}`.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--skip-steps", type=int, default=0, help="extra captured steps to drop from the front")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument(
        "--top-other",
        type=int,
        default=10,
        help="top exposed kernels per rank to explain with the waited-for rank's activity",
    )
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in a.trace_dir.iterdir() if re.match(r".*trace.*\.json(\.gz)?$", p.name))
    if not files:
        sys.exit(f"no trace files in {a.trace_dir}")

    lines = [f"# FSDP2 comm–comp overlap\n\nTraces: `{a.trace_dir}` ({len(files)} ranks)\n"]
    run_hta(a.trace_dir, a.out_dir, lines)

    all_steps, all_exposed, all_agg, all_tot, all_ctx = {}, {}, {}, {}, {}
    for p in files:
        tr = load_trace(p)
        r = rank_of(tr, p)
        s, e, g, t, c = analyze_rank(tr, a.skip_steps)
        all_steps[r], all_exposed[r], all_agg[r], all_tot[r], all_ctx[r] = s, e, g, t, c
        del tr
    wait = est_wait(all_exposed)
    skew = split_skew(all_exposed, all_ctx) if wait else None
    if skew:
        for r, rows in all_steps.items():
            for row in rows:
                sk, res = skew[r].get(row["step"], (float("nan"), float("nan")))
                row["skew_wait_exposed_ms"], row["residual_exposed_ms"] = sk, res

    lines.append("\n## Manual analysis — per rank (summed over analyzed steps)\n")
    lines.append(
        "| rank | window ms | compute ms | comm ms | overlap % | exposed comm ms | exposed % of window | idle ms | est. NCCL wait ms |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    summary = {}
    for r in sorted(all_tot):
        t = all_tot[r]
        ovp = 100 * t["overlap_ms"] / t["comm_ms"] if t["comm_ms"] else float("nan")
        exw = 100 * t["exposed_comm_ms"] / t["window_ms"] if t["window_ms"] else 0
        w = wait[r] / 1e3 if wait else float("nan")
        summary[r] = dict(t, overlap_pct=ovp, exposed_pct_of_window=exw, est_wait_ms=w)
        lines.append(
            f"| {r} | {t['window_ms']:.1f} | {t['compute_ms']:.1f} | {t['comm_ms']:.1f} | {ovp:.1f} | "
            f"{t['exposed_comm_ms']:.1f} | {exw:.1f} | {t['idle_ms']:.1f} | {w:.1f} |"
        )
    if wait is None:
        lines.append("\n(est. NCCL wait unavailable: NCCL kernel sequences don't line up across ranks.)")

    if skew:
        lines.append("\n## Exposed comm = rank-skew wait + residual (per rank, summed over analyzed steps)\n")
        lines.append(
            "Skew wait = exposed time inside the first `est. wait` µs of each NCCL kernel (spinning until "
            "the last rank arrives). Residual = the rest: exposed transfer, an upper bound because the "
            "fastest rank's duration can itself include waiting.\n"
        )
        lines.append("| rank | exposed comm ms | = skew wait ms | + residual ms | residual % of window |")
        lines.append("|---|---|---|---|---|")
        for r in sorted(all_tot):
            sk = sum(v[0] for v in skew[r].values())
            res = sum(v[1] for v in skew[r].values())
            win = all_tot[r]["window_ms"]
            summary[r].update(skew_wait_exposed_ms=sk, residual_exposed_ms=res)
            lines.append(
                f"| {r} | {all_tot[r]['exposed_comm_ms']:.1f} | {sk:.1f} | {res:.1f} | "
                f"{100 * res / win if win else 0:.1f} |"
            )

    lines.append("\n## Where exposed communication happens (rank 0; summed over steps)\n")
    r0 = min(all_agg)
    g = all_agg[r0]
    keys = sorted({k[:3] for k in g}, key=lambda k: -g[k + ("exposed",)])
    lines.append("| collective | phase | FSDP label | count | total ms | exposed ms | exposed % |")
    lines.append("|---|---|---|---|---|---|---|")
    for k in keys:
        tt, ex, n = g[k + ("total",)], g[k + ("exposed",)], g[k + ("count",)]
        lines.append(
            f"| {k[0]} | {k[1]} | `{k[2]}` | {n} | {tt / 1e3:.2f} | {ex / 1e3:.2f} | {100 * ex / tt if tt else 0:.0f} |"
        )

    lines.append(f"\n## Top {a.top} exposed NCCL kernels, rank {r0} (jump to these in Perfetto)\n")
    lines.append(
        "| offset ms | step | pos in step % | collective | phase | label | dur µs | exposed µs | est. wait µs |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for x in sorted(all_exposed[r0], key=lambda x: -x["exposed_us"])[: a.top]:
        lines.append(
            f"| {x['offset_ms_from_trace_start']} | {x['step']} | {x['pos_in_step_pct']} | {x['collective']} | "
            f"{x['phase']} | `{x['label'][:70]}` | {x['dur_us']} | {x['exposed_us']} | {x.get('est_wait_us', '')} |"
        )

    if skew and a.top_other > 0:
        lines.append("\n## What the waited-for rank was doing during the top exposed kernels\n")
        lines.append(
            "Same wall-clock window on the rank that launched the collective last (valid when ranks share a "
            "host clock, i.e. single node). Offsets are from each rank's own trace start. "
            "`other busy %` = share of the window with a compute kernel running on that rank.\n"
        )
        for r in sorted(all_exposed):
            lines.append(f"\n### rank {r}\n")
            lines.append(
                "| offset ms | step | collective | phase | label | exposed µs | skew µs | residual µs "
                "| waited-for rank | its offset ms | other busy % | its top GPU work | its top CPU ops |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for x in sorted(all_exposed[r], key=lambda x: -x["exposed_us"])[: a.top_other]:
                o = x.get("waited_for_rank")
                if o is None:
                    continue
                lo, hi = x["ts_us"], x["ts_us"] + x["dur_us"]
                busy, gpu, cpu = describe_window(all_ctx[o], lo, hi)
                lines.append(
                    f"| {x['offset_ms_from_trace_start']} | {x['step']} | {x['collective']} | {x['phase']} "
                    f"| `{x['label'][:60]}` | {x['exposed_us']} | {x['skew_exposed_us']} "
                    f"| {x['residual_exposed_us']} | {o} | {round((lo - all_ctx[o]['t0']) / 1e3, 3)} "
                    f"| {busy:.0f} | {gpu} | {cpu} |"
                )

    # machine-readable outputs
    rows = [{"rank": r, **row} for r in sorted(all_steps) for row in all_steps[r]]
    if rows:
        with open(a.out_dir / "per_step.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with open(a.out_dir / "nccl_kernels.csv", "w", newline="") as f:
        fields = [
            "rank",
            "step",
            "collective",
            "phase",
            "label",
            "stream",
            "dur_us",
            "exposed_us",
            "est_wait_us",
            "offset_ms_from_trace_start",
            "pos_in_step_pct",
            "corr",
            "kernel",
            "skew_exposed_us",
            "residual_exposed_us",
            "waited_for_rank",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(all_exposed):
            for x in all_exposed[r]:
                w.writerow({"rank": r, **x})
    (a.out_dir / "summary.json").write_text(json.dumps({str(k): v for k, v in summary.items()}, indent=2))
    (a.out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {a.out_dir}/summary.md, per_step.csv, nccl_kernels.csv, summary.json")


if __name__ == "__main__":
    main()
