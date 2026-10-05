# FSDP2 computation–communication overlap profiling

Baseline measurement for the modality-parallel comparison: how much of FSDP2's
communication (param all-gathers, grad reduce-scatters) is hidden behind compute,
and where the exposed part happens.

## Files
- `profile_fsdp_overlap.sh` — runs a Cosmos SFT recipe for a few iterations under the
  repo's built-in torch.profiler hook (`trainer.profiling.*`), then runs the analysis.
- `analyze_comm_overlap.py` — HTA `get_comm_comp_overlap()` / `get_temporal_breakdown()`
  plus an independent stdlib analysis that attributes exposed NCCL time to
  collective type, forward/backward/optimizer, and the FSDP2 label (with module FQN),
  splits exposed time into rank-skew wait vs. residual, and shows what the waited-for
  rank was running.
- `check_stall.py` — lists every GPU kernel (per stream) in one time window of one rank's
  trace: the scripted version of checking a stall in Perfetto.
- `per_step_inputs.py` — vision recipes only: per step and rank, dataloader wait, packed
  sequence layout (needs `trainer.profiling.record_shape=true`) and VAE-encode work, next to
  overlap and wait; attributes the largest waits to the other rank's extra work.

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
python tools/fsdp_overlap/per_step_inputs.py --trace-dir <.../torch_trace/iteration_N> --analysis-dir <out>
python tools/fsdp_overlap/check_stall.py --trace <.../rank0_trace.json.gz> --offset-ms <offset> --dur-ms <dur>
```

## Running on new hardware
**Prerequisites**
- Repo environment per `docs/setup.md`. Pick the CUDA group your driver supports
  (`nvidia-smi` "CUDA Version": ≥ 13.0 → `--group=cu130-train`, 12.8 → `--group=cu128-train`).
- Recipe data and checkpoints per `docs/training.md` Steps 1–2.
- `ffmpeg`/`ffprobe` on `PATH` (vision recipe; the decoder shells out to them). Use FFmpeg ≤ 7:
  `cosmos_framework/data/generator/local_datasets/helper.py` passes `-vsync`, which FFmpeg 8+
  rejects — every sample then silently decodes to zero frames ("No frames decoded… skipping").
- FFmpeg 4–8 *shared libraries* visible to the loader (videophy2 recipe; torchcodec decodes
  video). If the system has none, PyAV's bundled FFmpeg works: symlink every file in
  `site-packages/av.libs/` into one directory, add plain-soname links (`libavutil.so.60` →
  `libavutil-<hash>.so.60.*`, same for avcodec/avformat/avdevice/avfilter/swscale/swresample)
  and put that directory on `LD_LIBRARY_PATH`.
- ≥ 2 GPUs on **one node**. The script launches single-node `torchrun`, and the cross-rank parts
  of the analysis compare timestamps across ranks, which needs a shared host clock.
- Optional: `pip install HolisticTraceAnalysis` for the HTA cross-check.

**Environment variables**

| variable | default | meaning |
|---|---|---|
| `RECIPE` | `vision_edge` | `vision_edge` or `videophy2_edge` |
| `NPROC_PER_NODE` | all visible GPUs | ranks = FSDP shard degree |
| `DATASET_PATH`, `BASE_CHECKPOINT_PATH`, `WAN_VAE_PATH` | `examples/...` | vision_edge inputs |
| `VIDEOPHYSICS_ROOT` | — (required) | videophy2_edge data root |
| `OUTPUT_ROOT` | `outputs/fsdp_overlap` | where runs, traces and analysis go |
| `PROFILE_WAIT` / `PROFILE_WARMUP` / `PROFILE_ACTIVE` | 5 / 3 / 3 | profiler schedule |
| `EXTRA_OVERRIDES` | — | extra Hydra overrides; recorded in `run_info.txt` |
| `MASTER_PORT` | 50012 | torchrun port |
| `KEEP_CHECKPOINTS`, `SKIP_ANALYSIS` | 0 | keep the exit checkpoint / only collect traces |

`HF_HOME`, `PYTORCH_ALLOC_CONF`, `LD_LIBRARY_PATH` and `NCCL_*` are passed through;
`run_info.txt` records the last three.

**Picking `NPROC_PER_NODE`.** Use the same GPU count the modality-parallel run will use, so both
shard over the same number of ranks; more ranks means smaller shards, more collectives and a
different overlap picture. The recipes are sized for 8× 80 GB. On smaller GPUs you will
likely need `EXTRA_OVERRIDES` (vision_edge: `model.config.ema.enabled=false`, then lower
`model.config.max_num_tokens_after_packing` together with `dataloader_train.max_sequence_length`).
That changes the workload, so it must be reported. Keep the default schedule unless a later
iteration runs out of memory.

**What to send back**, from `$OUTPUT_ROOT/<recipe>_<N>gpu_<timestamp>/`:
- `analysis/summary.md` — results
- `run_info.txt` — GPUs, topology, versions, overrides and env (needed to interpret anything)
- `nccl_transport.txt` — NVLink/P2P vs SHM vs network
- optionally `analysis/per_step.csv` and `analysis/nccl_kernels.csv` (small). Traces are
  ~40 MB per rank per 8 steps; send them only if asked.

## Outputs (`outputs/fsdp_overlap/<recipe>_<N>gpu_<timestamp>/`)
- `run_info.txt` — GPUs, topology (`nvidia-smi topo -m`), torch/CUDA/NCCL versions, overrides, env
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

Added on top (the metrics above are unchanged):
- **exposed comm = skew wait + residual.** The first `est. NCCL wait` µs of each NCCL kernel are
  treated as waiting for the last rank to arrive; exposed time inside that span is *skew wait*,
  and the rest is *residual* (exposed transfer). Residual is an upper bound: the fastest rank's
  duration can itself contain some waiting. Per kernel these are new columns in
  `nccl_kernels.csv` (`skew_exposed_us`, `residual_exposed_us`, `waited_for_rank`) and per step
  in `per_step.csv` (`skew_wait_exposed_ms`, `residual_exposed_ms`).
- **What the waited-for rank was doing.** For each rank's top exposed kernels (`--top-other`,
  default 10): the rank that launched the same collective last, its GPU-busy % in the same
  wall-clock window, its top GPU work (kernels launched under `aten::conv3d` are labeled
  `conv3d (VAE encode)`) and its top CPU ops (e.g. `enumerate(DataLoader)`, `aten::item` host
  syncs).
- **HTA, GPU-only.** With `trainer.profiling.record_shape=true`, Inductor's Triton launch ops
  (CPU events) carry a `stream` arg and HTA counts them as GPU compute, inflating overlap by
  about 4 points. The unfiltered HTA output is kept, and a second HTA result with those rows
  removed is added.

Note: the bare `FSDP::all_gather` label (no module name) is the **root FSDP unit**, i.e. every
parameter outside the decoder blocks (for Cosmos3-Edge: token embedding + LM head + small
projections, ≈1 GiB bf16 unsharded). FSDP2 appends the FQN only when it is non-empty.

## Manual confirmation in a trace viewer
Open a `rank0_trace.json.gz` in https://ui.perfetto.dev (chrome://tracing chokes on large traces).
Take a row from "Top exposed NCCL kernels" in `summary.md`; its `offset ms` is measured from the
first event in the trace. Check that the NCCL stream has a kernel there and every compute
stream on that GPU is empty directly underneath it. A low overlap % with no such gaps means the
metric is being fooled; gaps with no NCCL kernel mean the stall is something else
(CPU/launch-bound, dataloader, host sync). `check_stall.py` does the same check from the
command line.

## Caveats
- Numbers are hardware-specific. Consumer GPUs without NVLink/P2P (e.g. RTX 4090) route NCCL
  through host memory; exposure will be far worse than on NVLink systems. Always compare runs
  with `run_info.txt` + `nccl_transport.txt` alongside.
- The trainer writes a final checkpoint at exit; the script deletes it unless `KEEP_CHECKPOINTS=1`.
- If you OOM and add `EXTRA_OVERRIDES` (smaller token budget, EMA off), you changed the workload.
  Record it.
- videophy2_edge (release 2026-09-23) failed here before training for a reason unrelated to the
  hardware: `hf_model.py:542` drops `attention_mask` only for `nemotron_vl`/`nemotron_siglip2`,
  not `cosmos3_edge`, so mRoPE positions come out shorter than the 64-padded sequence
  ("size of tensor a (1664) must match … (1659)"). Adding `cosmos3_edge` to that set got past it.
  For this recipe `dataloader_train.max_sequence_length` does not cap per-sample memory: an
  over-long sample is emitted alone at full length.
