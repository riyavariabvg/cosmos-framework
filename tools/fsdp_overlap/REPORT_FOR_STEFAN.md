# FSDP2 compute–communication overlap: Cosmos3-Edge baseline

*Riya Varia, 2026-10-05. Tools: `tools/fsdp_overlap/` on branch `riya/fsdp2-overlap-profiling`.*

## Summary
- **vision_edge (Cosmos3-Edge vision SFT), 2× RTX 4090:** FSDP2 hides **76.5% (rank 0) / 86.2% (rank 1)**
  of communication behind compute. Exposed communication is **11.2% / 5.6%** of GPU time.
- **Most of rank 0's exposure is rank skew, not bandwidth.** The waiting rank sits in NCCL while the
  other rank is still encoding its videos (VAE). Packed token counts are balanced across ranks, but
  VAE work is not.
- **Exposure from actual data transfer is ~4–5% of each step on both ranks**, concentrated at the
  edges of the FSDP pipeline where there is nothing to prefetch behind. The largest piece is the
  **root FSDP unit's ~1 GiB all-gather at the start of every forward** (~84 ms each, fully exposed).
- **HTA agrees with the manual analysis to within 0.2 points** once two HTA input issues are corrected
  (details below).
- **videophy2_edge was not measured.** It hit a repo bug (fixable) and then ran out of memory in a
  way the approved fixes cannot address; see open questions.

All numbers are from 2 consumer GPUs over PCIe with no GPU-to-GPU path, at a reduced token budget.
Treat them as a methods check and a lower bound on how well FSDP2 hides communication, not as the
number to compare against an NVLink machine.

## Hardware and software
- 2× RTX 4090 24 GB, one node. Topology `PHB` (both GPUs behind the CPU's PCIe host bridge; no NVLink, no P2P).
- NCCL 2.27.5 picked **`SHM/direct`**, i.e. GPU↔GPU traffic goes through host memory (`nccl_transport.txt`).
- torch 2.10.0+cu128, driver 570 (CUDA 12.8), NCCL 2.27.5, HTA 0.5.0.

## vision_edge

**Run:** `/scratch/riyavaria/fsdp_overlap_out/vision_edge_2gpu_20260929_215101/` (`run_info.txt`, `deviations.txt`,
`analysis_v2/summary.md`, `analysis_v2/per_step_inputs.md`).

**Deviations from the stock recipe** (`examples/toml/sft_config/vision_sft_edge.toml`), all to fit 24 GB:

| setting | stock | used |
|---|---|---|
| `model.config.max_num_tokens_after_packing` / `dataloader_train.max_sequence_length` | 45056 | **12288** |
| `model.config.ema.enabled` | true | **false** |
| `PYTORCH_ALLOC_CONF` | unset | `expandable_segments:True` |
| profiler schedule (wait / warmup / active) | 5 / 3 / 3 | **1 / 1 / 8** (8 captured steps for statistics) |
| `trainer.profiling.record_shape` | false | **true** (to read packed lengths) |
| GPUs | 8× H100 | 2× RTX 4090 |

Captured iterations 3–10 are all steady state, 6.9–7.3 s each. torch.compile ran in iteration 1 (105 s),
which was skipped. The stock 45,056 and the 32,768 / 16,384 budgets all ran out of memory on 24 GB.

**Results (summed over the 8 captured steps, 56.5 s of GPU time per rank)**

| | rank 0 | rank 1 |
|---|---|---|
| overlap % (manual) | **76.5** | **86.2** |
| overlap % (HTA, after corrections) | 76.32 | 86.01 |
| exposed comm | 6,326 ms (11.2%) | 3,185 ms (5.6%) |
| … of which rank-skew wait | 3,583 ms | 851 ms |
| … of which residual (exposed transfer) | 2,743 ms (4.9%) | 2,334 ms (4.1%) |
| GPU idle (nothing running) | 1,909 ms (3.4%) | 2,116 ms (3.7%) |

*Skew wait* is the part of exposed time inside each NCCL kernel's first "est. wait" µs (kernel duration
minus the fastest rank's duration for the same collective). The rest is *residual*. The residual is an
upper bound on true exposed transfer, because the fastest rank's duration can itself include some waiting.

**Where the residual exposure happens** (each micro-batch, so twice per optimizer step; positions are
% of the step):

| collective | phase | FSDP unit | position in step | residual each (rank 0 / 1) |
|---|---|---|---|---|
| AllGather | forward start | **root** (`FSDP::all_gather`, no module name) | ~20% and ~72% | 84 / 83 ms |
| AllGather | backward start | `layers.27` (last decoder layer) | ~33% and ~84% | 15 / 15 ms |
| AllGather | forward | `layers.0` | ~26% and ~76% | 9 / 9 ms |
| ReduceScatter | backward end | `layers.0` | ~49% and ~99% | 6 / 5 ms |

These four account for 67% (rank 0) / 77% (rank 1) of the residual. The rest is spread thinly over the other
layers. This is the expected FSDP pattern: the first gather of the forward, the first gather of the
backward, and the last reduce-scatter have no compute to overlap with.

**The root FSDP unit.** The bare `FSDP::all_gather` label belongs to the root unit. FSDP2 appends the module
FQN only when it is non-empty (`torch/.../_fsdp_param_group.py:804`), and the root is wrapped at
`cosmos_framework/model/generator/mot/parallelize_vfm_network.py:144` with FQN `""`. The per-layer units
are the 28 `language_model.model.layers.N` blocks (`parallelize_unified_mot.py:519`). From the trace, the
root all-gather moves **542.4 M bf16 elements = 1,034.5 MiB** (5.4× one decoder layer's 192 MiB), once per
micro-batch forward. From the DCP checkpoint metadata, the root owns everything outside the decoder blocks:
`embed_tokens` 268.4 M + `lm_head` 268.4 M (both gathered; not tied in FSDP) + `time_embedder`,
`vae2llm`/`llm2vae`, norms. The 8.5 M action-head parameters are not in this model.

**Skew comes from uneven VAE work, not from FSDP.** Per step and rank (`per_step_inputs.md`):
- Packed micro-batches are balanced, at 10.3k–12.3k tokens against the 12,288 budget.
- Dataloader wait is ~1 ms per step and can be ignored.
- VAE-encode GPU time differs: rank 0 does 390–490 ms less of it in steps 3, 4, 5 and 7, and those are
  exactly its 0.9–1.3 s wait steps. In 7 of 8 steps, the rank with less VAE work is the one that waits.
- For the 8 largest waits, the other rank's extra compute before that collective matches the wait to
  within 3–8% (e.g. 571 ms wait vs 544 ms extra), and about two-thirds of that extra compute is VAE `conv3d`.
- The packer balances language-model tokens, not raw video frames. The VAE runs unsharded on each rank,
  so its cost per rank varies with how much video each rank packed.

Not explained: in 2 of the 10 largest rank-0 waits, rank 1 was only ~15% busy, and its CPU spent
~168 ms in `aten::empty` inside its own `FSDP::all_gather`. That looks like an allocator stall near the 24 GB
limit, but this was not verified.

**HTA vs. manual.** They now agree within 0.2 points per rank. Two HTA issues had to be worked around; the raw
HTA output is still reported alongside:
1. `TraceAnalysis` defaults to `include_last_profiler_step=False` and drops the last captured step's kernels.
   The analyzer now passes `True`. Before this, a 2-step capture reported 86.5% vs 72.7% manual.
2. With `record_shape=true`, Inductor's Triton launch ops (CPU events) carry a `stream` field, and HTA counts
   them as GPU compute: 80.73 / 90.74% raw vs 76.32 / 86.01% with those rows removed.

**Manual check in Perfetto (prepared, scripted check done).** Trace files:
`.../vision_edge_2gpu_20260929_215101/imaginaire/cosmos3/sft/vision_sft_edge/torch_trace/iteration_10/rank{0,1}_trace.json.gz`.
Offsets are ms from each trace's first event; rank 1's trace starts 0.065 ms earlier, hence the +0.065.
Streams: 7 = compute, 22 = NCCL all-gather, 18 = FSDP all-gather copies.

| stall | rank 0 offset | rank 1 offset | what you should see | `check_stall.py` result |
|---|---|---|---|---|
| A. skew-dominated root forward AllGather (step 5) | 25576.923, 655.6 ms | 25576.988 (same window); its own AllGather starts at 26147.976 | rank 0: one long NCCL AllGather on stream 22, compute streams empty. Rank 1: dense VAE conv + elementwise kernels on stream 7, then its 84.6 ms AllGather at the end | rank 0: NCCL 100%, compute 0.1%. Rank 1: compute 83%, NCCL 12.9% |
| B. ~84 ms residual root forward AllGather (step 5, micro-batch 1) | 22496.255, 83.6 ms | 22496.320, 83.6 ms | both ranks: one NCCL AllGather, nothing on any compute stream | rank 0: compute 0.0%. Rank 1: compute 0.8% |
| C. `layers.27` backward AllGather (step 2) | 2365.335, 24.2 ms | 2365.400 | rank 0: 24.2 ms AllGather with only small elementwise kernels underneath. Rank 1: its AllGather starts ~7 ms later | rank 0: NCCL 100%, compute 11.8%. Rank 1: compute 36.5% |

Reproduce with e.g. `python tools/fsdp_overlap/check_stall.py --trace <rank0_trace.json.gz> --offset-ms 25576.923 --dur-ms 655.6`.

## videophy2_edge: not measured
Data prepared per `docs/training.md` (3,343 train / 3,397 val clips, 0 download failures). Three attempts:
1. **Environment:** the dataloader's `torchcodec` needs FFmpeg shared libraries. Fixed by exposing PyAV's
   bundled FFmpeg 8 libraries via `LD_LIBRARY_PATH`; no workload change.
2. **Repo bug:** `size of tensor a (1664) must match … (1659)` in rotary embeddings.
   - VLM batches are right-padded to a multiple of 64 (`collate_fn.py:130`), and `get_rope_index` drops
     padded positions when it gets an `attention_mask`.
   - `hf_model.py:542` already removes the mask for this reason, but only for `nemotron_vl`/`nemotron_siglip2`,
     not `cosmos3_edge`.
   - Adding `cosmos3_edge` there (a temporary local patch, reverted; diff in `fsdp_overlap_out/`) got past it.
     As written, this recipe should fail on any hardware.
3. **Out of memory in the first backward:** tried to allocate 4.93 GiB with 3.72 GiB free. That size is consistent
   with the LM-head logits of a ~10k-token sample.
   - The approved fixes don't apply: EMA is already off in this recipe, and `max_sequence_length` cannot cap
     per-sample memory here, because an over-long sample is emitted alone at full length (`batchers.py:114`)
     and `max_samples_per_batch = 1`.
   - I stopped at 3 of 4 attempts rather than spend the last one on a change that cannot help.

## Caveats
- **Hardware:** PCIe with NCCL over host memory, so transfers are far slower than over NVLink. The ~84 ms root
  gather would be much shorter on an NVLink system (not measured), and the residual share would drop with it.
- **2 GPUs only:** shards are half the model. With 8 ranks the shards are smaller and there are more
  collectives, which gives a different picture.
- **Workload:** 12,288 tokens per micro-batch (27% of the recipe's 45,056) and EMA off. Less compute per step
  probably means less to hide communication behind, so overlap at the stock budget may be higher; untested.
- **Sample size:** the skew findings come from 8 steps on one data order. The direction (which rank waits)
  depends on which videos each rank draws.
- `record_shape=true` adds CPU overhead per op. Its effect on step time was not measured (there is no 12,288-token run without it).
- The skew/residual split and the "waited-for rank" view compare timestamps across ranks. That is valid on one
  node only.

## Tool changes in this round
- `analyze_comm_overlap.py`: the skew-wait / residual split (per rank, per step, per kernel); "what the waited-for
  rank was doing" for the top exposed kernels; HTA rerun without stream-tagged CPU ops. Existing metrics are
  unchanged: old CSV columns were verified identical on the existing traces.
- New: `check_stall.py` (scripted Perfetto check) and `per_step_inputs.py` (per-step input balance, vision recipes).
- `profile_fsdp_overlap.sh`: `run_info.txt` now also records `PYTORCH_ALLOC_CONF`, `LD_LIBRARY_PATH` and `NCCL_*`.
- README: "Running on new hardware" (prerequisites incl. the FFmpeg ≤ 7 / torchcodec requirements, env vars,
  choosing `NPROC_PER_NODE`, what to send back). `bash -n` passes; all tools were rerun on existing traces.

## Open questions for Riya
1. **videophy2_edge:** should I reduce the per-sample video size (fewer frames or lower resolution; not in the approved
   OOM order), run on bigger GPUs, or drop it from the comparison?
2. **The `hf_model.py` fix for `cosmos3_edge`:** report it to NVIDIA / upstream it, or keep it local? It is not committed.
3. **Default schedule:** I did not rerun vision_edge with 5/3/3, because the existing 1/1/8 capture contains no compile
   warmup (the condition in the task). Do you want a 5/3/3 run anyway, for a like-for-like comparison with Stefan's runs?
4. **`aten::empty` stalls:** worth chasing (memory snapshot) or out of scope?
5. `CLAUDE.md` (the run-reporting rule) is still uncommitted. Commit it to this branch?
