# FSDP2 computation–communication overlap profiling

Baseline measurement for the modality-parallel comparison: how much of FSDP2's
communication (param all-gathers, grad reduce-scatters) is hidden behind compute,
and where the exposed part happens.

## Files
- `profile_fsdp_overlap.sh` — runs a Cosmos SFT recipe for a few iterations under the
  repo's built-in torch.profiler hook (`trainer.profiling.*`), then runs the analysis.
- `analyze_comm_overlap.py` — HTA `get_comm_comp_overlap()` / `get_temporal_breakdown()`
  plus an independent stdlib analysis that attributes exposed NCCL time to
  collective type, forward/backward/optimizer, and the FSDP2 label (with module FQN).

## Run
From the repo root, inside the training environment (same prerequisites as
`docs/training.md` Steps 1–2 for the chosen recipe):

```bash
pip install HolisticTraceAnalysis            # optional; manual analysis runs without it

# Recipe 1 (primary)
RECIPE=vision_edge NPROC_PER_NODE=<gpus> bash tools/fsdp_overlap/profile_fsdp_overlap.sh

# Recipe 2
RECIPE=videophy2_edge NPROC_PER_NODE=<gpus> VIDEOPHYSICS_ROOT=<dir> \
  bash tools/fsdp_overlap/profile_fsdp_overlap.sh
```

Default schedule: 5 skipped iterations (torch.compile, NCCL init), 3 profiler-warmup
iterations (discarded), 3 captured iterations. Change with `PROFILE_WAIT/WARMUP/ACTIVE`.
One profiled "step" = one optimizer iteration = `grad_accum_iter` forward/backward passes.

Re-analyze existing traces without re-running:
```bash
python tools/fsdp_overlap/analyze_comm_overlap.py --trace-dir <.../torch_trace/iteration_N> --out-dir <out>
```

## Outputs (`outputs/fsdp_overlap/<recipe>_<N>gpu_<timestamp>/`)
- `run_info.txt` — GPUs, topology (`nvidia-smi topo -m`), torch/CUDA/NCCL versions, overrides
- `nccl_transport.txt` — which transport NCCL used (P2P/NVLink vs SHM vs network)
- `analysis/summary.md` — main result tables
- `analysis/per_step.csv`, `analysis/nccl_kernels.csv`, `analysis/summary.json`, `analysis/hta_*.csv`
- the raw traces under `imaginaire/.../torch_trace/iteration_N/rank*_trace.json.gz`

## Metrics
Per rank over the captured GPU window:
- **overlap %** = |comm ∩ compute| / |comm| (the quantity HTA reports)
- **exposed comm** = comm time with no compute kernel running anywhere on that GPU → the stall
- **idle** = GPU running nothing at all
- **est. NCCL wait** = NCCL kernel duration minus the fastest rank's duration for the same
  collective. NCCL kernels spin while waiting for peers, so "exposed communication" can be
  rank skew (e.g. unequal packed-sequence lengths) rather than bandwidth. Report both.

## Manual confirmation in a trace viewer
Open a `rank0_trace.json.gz` in https://ui.perfetto.dev (chrome://tracing chokes on large traces).
Take a row from "Top exposed NCCL kernels" in `summary.md`; its `offset ms` is measured from the
first event in the trace. Check that the NCCL stream has a kernel there and every compute
stream on that GPU is empty directly underneath it. A low overlap % with no such gaps means the
metric is being fooled; gaps with no NCCL kernel mean the stall is something else
(CPU/launch-bound, dataloader, host sync).

## Caveats
- Numbers are hardware-specific. Consumer GPUs without NVLink/P2P (e.g. RTX 4090) route NCCL
  through host memory; exposure will be far worse than on NVLink systems. Always compare runs
  with `run_info.txt` + `nccl_transport.txt` alongside.
- The trainer writes a final checkpoint at exit; the script deletes it unless `KEEP_CHECKPOINTS=1`.
- If you OOM and add `EXTRA_OVERRIDES` (smaller token budget, EMA off), you changed the workload.
  Record it.
