#!/usr/bin/env python3
"""Per-step, per-rank input balance for FSDP2 overlap traces.

Answers: is exposed NCCL time caused by FSDP communication, or by ranks arriving at a
collective at different times because they packed different amounts of work?

Requires traces captured with `trainer.profiling.record_shape=true` (for packed lengths)
and the outputs of analyze_comm_overlap.py (per_step.csv, nccl_kernels.csv) in --analysis-dir.

Per rank and profiler step it reports:
  - overlap %, exposed comm, est. NCCL wait        (from analyze_comm_overlap.py)
  - dataloader wait: count and total CPU time of DataLoader.__next__
  - per micro-batch packed layout, read from natten::fmha_forward input shapes
    (two-way MoT attention: und self-attention has Q == K; gen attention has
    Q = gen tokens, K = und + gen tokens = packed length; cu_seqlens length - 1 = samples)
  - VAE encode: aten::conv3d count and GPU time of the kernels it launched

and, for the largest est. NCCL waits, the per-rank compute issued between the previous
collective and that one (and how much of the difference is conv3d / VAE).

Usage:
  python per_step_inputs.py --trace-dir <.../torch_trace/iteration_N> --analysis-dir <run>/analysis
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import re
import sys
from pathlib import Path

from analyze_comm_overlap import is_nccl, load_trace, merge, rank_of, total

LAUNCH_CATS = ("cuda_runtime", "cuda_driver")


def ranges_by_thread(events):
    out = collections.defaultdict(list)
    for e in events:
        out[(e["pid"], e["tid"])].append((e["ts"], e["ts"] + e["dur"]))
    return {k: sorted(v) for k, v in out.items()}


def inside(ranges, ts):
    """ts falls inside one of the sorted (possibly nested) ranges."""
    i = bisect.bisect_right(ranges, (ts, float("inf"))) - 1
    while i >= 0:
        s, e = ranges[i]
        if s <= ts < e:
            return True
        if e < ts and i > 0 and ranges[i - 1][1] < s:
            break
        i -= 1
    return False


def analyze_rank(trace):
    ev = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X" and "dur" in e]
    steps = sorted(
        (
            (int(e["name"].split("#")[1]), e["ts"], e["ts"] + e["dur"])
            for e in ev
            if e.get("cat") == "user_annotation" and e["name"].startswith("ProfilerStep#")
        ),
        key=lambda x: x[1],
    )

    def step_of(ts):
        for sid, s, e in steps:
            if s <= ts < e:
                return sid
        return None

    launch = {}
    for e in ev:
        if e.get("cat") in LAUNCH_CATS and "correlation" in e.get("args", {}):
            launch[e["args"]["correlation"]] = (e["ts"], (e["pid"], e["tid"]))

    rows = {sid: dict(dl_calls=0, dl_ms=0.0, conv3d_calls=0, vae_conv_gpu_ms=0.0, mbs=[]) for sid, _, _ in steps}

    for e in ev:
        if e["name"].startswith("enumerate(DataLoader)"):
            sid = step_of(e["ts"])
            if sid in rows:
                rows[sid]["dl_calls"] += 1
                rows[sid]["dl_ms"] += e["dur"] / 1e3

    # VAE encode proxy: kernels launched from inside aten::conv3d (the LM has no 3D convs).
    conv = [e for e in ev if e.get("cat") == "cpu_op" and e["name"] == "aten::conv3d"]
    for e in conv:
        sid = step_of(e["ts"])
        if sid in rows:
            rows[sid]["conv3d_calls"] += 1
    conv_ranges = ranges_by_thread(conv)
    kernels = [e for e in ev if e.get("cat") == "kernel"]
    conv_kernel_ids = set()
    for k in kernels:
        c = k.get("args", {}).get("correlation")
        if c in launch and not is_nccl(k["name"]):
            ts, th = launch[c]
            if th in conv_ranges and inside(conv_ranges[th], ts):
                conv_kernel_ids.add(id(k))
                sid = step_of(ts)
                if sid in rows:
                    rows[sid]["vae_conv_gpu_ms"] += k["dur"] / 1e3

    # Micro-batch boundaries: one LM token-embedding lookup (largest vocab table) per micro-batch.
    emb = [e for e in ev if e["name"] == "aten::embedding" and e.get("args", {}).get("Input Dims")]
    vocab = max((e["args"]["Input Dims"][0][0] for e in emb if e["args"]["Input Dims"][0]), default=None)
    bounds = sorted(e["ts"] for e in emb if e["args"]["Input Dims"][0] and e["args"]["Input Dims"][0][0] == vocab)
    fmha = sorted(
        (e for e in ev if e["name"] == "natten::fmha_forward" and e.get("args", {}).get("Input Dims")),
        key=lambda e: e["ts"],
    )
    for i, b in enumerate(bounds):
        end = bounds[i + 1] if i + 1 < len(bounds) else float("inf")
        sid = step_of(b)
        if sid not in rows:
            continue
        und = gen = None
        for f in fmha:
            if not b <= f["ts"] < end:
                continue
            d = f["args"]["Input Dims"]
            q, k = d[0][1], d[1][1]
            n = d[7][0] - 1 if len(d) > 7 and d[7] else None
            if q == k and und is None:
                und = (q, n)
            elif q < k and gen is None:
                gen = (q, k, n)
            if und and gen:
                break
        rows[sid]["mbs"].append(
            dict(
                und_tokens=und[0] if und else None,
                gen_tokens=gen[0] if gen else None,
                packed_len=gen[1] if gen else (und[0] if und else None),
                samples=gen[2] if gen else (und[1] if und else None),
            )
        )

    nccl = sorted(
        (k for k in kernels if is_nccl(k["name"])),
        key=lambda k: (k.get("args", {}).get("correlation") is None, k.get("args", {}).get("correlation", 0)),
    )
    comp = [k for k in kernels if not is_nccl(k["name"])]
    return rows, nccl, comp, conv_kernel_ids


def work_between(comp, conv_ids, lo, hi):
    ks = [k for k in comp if lo <= k["ts"] < hi]
    busy = total(merge([[k["ts"], k["ts"] + k["dur"]] for k in ks])) / 1e3
    vae = total(merge([[k["ts"], k["ts"] + k["dur"]] for k in ks if id(k) in conv_ids])) / 1e3
    return busy, vae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True, type=Path)
    ap.add_argument("--analysis-dir", required=True, type=Path, help="analyze_comm_overlap.py --out-dir")
    ap.add_argument("--top", type=int, default=8, help="largest est. NCCL waits to attribute")
    a = ap.parse_args()

    files = sorted(p for p in a.trace_dir.iterdir() if re.match(r".*trace.*\.json(\.gz)?$", p.name))
    if not files:
        sys.exit(f"no trace files in {a.trace_dir}")

    per_step = {(int(r["rank"]), int(r["step"])): r for r in csv.DictReader(open(a.analysis_dir / "per_step.csv"))}
    waits = collections.Counter()
    top_waits = collections.defaultdict(list)
    for r in csv.DictReader(open(a.analysis_dir / "nccl_kernels.csv")):
        if r["est_wait_us"] != "":
            w = float(r["est_wait_us"])
            waits[(int(r["rank"]), int(r["step"]))] += w
            top_waits[int(r["rank"])].append((w, r))
    if not waits:
        print("note: nccl_kernels.csv has no est_wait_us (sequences did not line up across ranks)")

    data = {}
    for p in files:
        tr = load_trace(p)
        data[rank_of(tr, p)] = analyze_rank(tr)
        del tr
    ranks = sorted(data)

    lines = [f"# Per-step input balance\n\nTraces: `{a.trace_dir}`\n"]
    lines.append(
        "Packed layout per micro-batch: `packed = und + gen tokens (samples)`. Trainer iteration = profiler step + 1.\n"
    )
    lines.append(
        "| step | rank | overlap % | exposed comm ms | est. NCCL wait ms | dataloader calls | dataloader ms "
        "| micro-batches: packed = und + gen (samples) | conv3d calls | VAE conv GPU ms |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    rows_out = []
    for sid in sorted({s for r in ranks for s in data[r][0]}):
        for r in ranks:
            row = data[r][0].get(sid)
            if row is None:
                continue
            ps = per_step.get((r, sid), {})
            mb = "; ".join(
                f"{m['packed_len']} = {m['und_tokens']} + {m['gen_tokens']} ({m['samples']})" for m in row["mbs"]
            )
            ov = float(ps["overlap_pct"]) if ps else float("nan")
            ex = float(ps["exposed_comm_ms"]) if ps else float("nan")
            w = waits[(r, sid)] / 1e3
            lines.append(
                f"| {sid} | {r} | {ov:.1f} | {ex:.1f} | {w:.1f} | {row['dl_calls']} | {row['dl_ms']:.1f} "
                f"| {mb} | {row['conv3d_calls']} | {row['vae_conv_gpu_ms']:.1f} |"
            )
            rows_out.append(
                dict(
                    step=sid,
                    rank=r,
                    overlap_pct=round(ov, 2),
                    exposed_comm_ms=round(ex, 1),
                    est_wait_ms=round(w, 1),
                    dataloader_calls=row["dl_calls"],
                    dataloader_ms=round(row["dl_ms"], 2),
                    microbatches=len(row["mbs"]),
                    packed_len=sum(m["packed_len"] or 0 for m in row["mbs"]),
                    und_tokens=sum(m["und_tokens"] or 0 for m in row["mbs"]),
                    gen_tokens=sum(m["gen_tokens"] or 0 for m in row["mbs"]),
                    samples=sum(m["samples"] or 0 for m in row["mbs"]),
                    conv3d_calls=row["conv3d_calls"],
                    vae_conv_gpu_ms=round(row["vae_conv_gpu_ms"], 1),
                )
            )

    # Attribute the largest waits: what did each rank run between the previous collective and this one?
    if len(ranks) == 2 and waits:
        lines.append(f"\n## Largest est. NCCL waits: work issued since the previous collective\n")
        lines.append(
            "A waiting rank arrived early. If the other rank's extra compute ≈ the wait, the wait is "
            "input/work imbalance, not communication.\n"
        )
        lines.append(
            "| waiting rank | step | collective | label | wait ms | compute since prev. collective: waiter / other ms "
            "| other − waiter ms | of which conv3d (VAE) ms |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        seqs = {r: data[r][1] for r in ranks}
        corr_index = {r: {k["args"].get("correlation"): i for i, k in enumerate(seqs[r])} for r in ranks}
        cand = sorted(((w, r, row) for r in ranks for w, row in top_waits[r]), key=lambda x: -x[0])[: a.top]
        for w, r, row in cand:
            o = [x for x in ranks if x != r][0]
            i = corr_index[r].get(int(row["corr"])) if row["corr"] else None
            if i is None or i == 0 or i >= len(seqs[o]):
                continue
            res = {}
            for rr in (r, o):
                prev, cur = seqs[rr][i - 1], seqs[rr][i]
                res[rr] = work_between(data[rr][2], data[rr][3], prev["ts"] + prev["dur"], cur["ts"])
            d_busy = res[o][0] - res[r][0]
            d_vae = res[o][1] - res[r][1]
            lines.append(
                f"| {r} | {row['step']} | {row['collective']} {row['phase']} | `{row['label'][:50]}` | {w / 1e3:.1f} "
                f"| {res[r][0]:.1f} / {res[o][0]:.1f} | {d_busy:.1f} | {d_vae:.1f} |"
            )

    out_csv = a.analysis_dir / "per_step_inputs.csv"
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        wr.writeheader()
        wr.writerows(rows_out)
    (a.analysis_dir / "per_step_inputs.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out_csv} and per_step_inputs.md")


if __name__ == "__main__":
    main()
