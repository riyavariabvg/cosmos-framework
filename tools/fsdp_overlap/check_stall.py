#!/usr/bin/env python3
"""Check what one rank's GPU was running in a time window (the scripted version of looking in Perfetto).

Lists every GPU kernel overlapping the window, per stream, and how much of the window had a
compute (non-NCCL) kernel running. An exposed-communication stall should show an NCCL kernel
and ~0% compute coverage.

The window can be given as an offset from the trace's first event (the `offset ms` columns in
summary.md) or as an absolute profiler timestamp in µs (comparable across ranks on one node).

Usage:
  python check_stall.py --trace rank0_trace.json.gz --offset-ms 25576.923 --dur-ms 655
  python check_stall.py --trace rank1_trace.json.gz --ts-us 7762724043.6 --dur-ms 79
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

from analyze_comm_overlap import is_nccl, kernel_label, load_trace, merge, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--offset-ms", type=float, help="window start, ms from the trace's first event")
    g.add_argument("--ts-us", type=float, help="window start, absolute profiler timestamp (µs)")
    ap.add_argument("--dur-ms", type=float, required=True)
    ap.add_argument("--top", type=int, default=5, help="kernels to list per stream")
    a = ap.parse_args()

    ev = [e for e in load_trace(a.trace).get("traceEvents", []) if e.get("ph") == "X" and "dur" in e]
    t0 = min(e["ts"] for e in ev)
    lo = a.ts_us if a.ts_us is not None else t0 + a.offset_ms * 1e3
    hi = lo + a.dur_ms * 1e3
    print(
        f"{a.trace.name}: window ts {lo:.1f}–{hi:.1f} µs = offset {(lo - t0) / 1e3:.3f}–{(hi - t0) / 1e3:.3f} ms "
        f"({a.dur_ms} ms)"
    )

    ks = [e for e in ev if e.get("cat") == "kernel" and e["ts"] < hi and e["ts"] + e["dur"] > lo]
    by_stream = collections.defaultdict(list)
    for k in ks:
        by_stream[k.get("tid")].append(k)
    for stream in sorted(by_stream, key=str):
        kk = by_stream[stream]
        cov = total(merge([[max(k["ts"], lo), min(k["ts"] + k["dur"], hi)] for k in kk])) / 1e3
        print(f"  stream {stream}: {len(kk)} kernels, covering {cov:.1f} ms of the window")
        names = collections.Counter()
        for k in kk:
            names[kernel_label(k["name"])] += min(k["ts"] + k["dur"], hi) - max(k["ts"], lo)
        for n, d in names.most_common(a.top):
            print(f"      {d / 1e3:9.2f} ms  {n}")
    comp = total(merge([[max(k["ts"], lo), min(k["ts"] + k["dur"], hi)] for k in ks if not is_nccl(k["name"])]))
    comm = total(merge([[max(k["ts"], lo), min(k["ts"] + k["dur"], hi)] for k in ks if is_nccl(k["name"])]))
    win = hi - lo
    print(f"  => NCCL running {100 * comm / win:.1f}% of window; compute (non-NCCL) running {100 * comp / win:.1f}%")


if __name__ == "__main__":
    main()
