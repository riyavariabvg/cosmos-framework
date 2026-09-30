#!/usr/bin/env bash
# Profile FSDP2 computation–communication overlap for a Cosmos SFT recipe.
#
# Runs cosmos_framework.scripts.train under the repo's built-in torch.profiler
# hook (cosmos_framework/utils/profiling.py), captures ACTIVE iterations after
# WAIT (skipped, covers torch.compile / NCCL init) + WARMUP (profiled but
# discarded) iterations, then runs analyze_comm_overlap.py on the traces.
#
# Usage (from repo root):
#   RECIPE=vision_edge     NPROC_PER_NODE=2 bash tools/fsdp_overlap/profile_fsdp_overlap.sh
#   RECIPE=videophy2_edge  NPROC_PER_NODE=2 VIDEOPHYSICS_ROOT=/path bash tools/fsdp_overlap/profile_fsdp_overlap.sh
#
# Env knobs (all optional unless stated):
#   RECIPE             vision_edge (default) | videophy2_edge
#   NPROC_PER_NODE     GPUs to use (default: all visible)
#   PROFILE_WAIT       iterations skipped entirely before warmup (default 5)
#   PROFILE_WARMUP     profiler warmup iterations, not analyzed (default 3)
#   PROFILE_ACTIVE     iterations captured and analyzed (default 3)
#   OUTPUT_ROOT        where training outputs + traces go (default outputs/fsdp_overlap)
#   EXTRA_OVERRIDES    extra Hydra-style overrides, space-separated, e.g.
#                      "model.config.max_num_tokens_after_packing=16384 dataloader_train.max_sequence_length=16384"
#                      (use only if you OOM; it changes what you're measuring — record it)
#   SKIP_ANALYSIS=1    only collect traces
#   vision_edge:   DATASET_PATH, BASE_CHECKPOINT_PATH, WAN_VAE_PATH (same defaults as launch_sft_vision_edge.sh)
#   videophy2_edge: VIDEOPHYSICS_ROOT (required), VLM_SAFETENSORS_PATH (optional)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RECIPE="${RECIPE:-vision_edge}"
PROFILE_WAIT="${PROFILE_WAIT:-5}"
PROFILE_WARMUP="${PROFILE_WARMUP:-3}"
PROFILE_ACTIVE="${PROFILE_ACTIVE:-3}"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then NPROC_PER_NODE=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | grep -c .)
  else NPROC_PER_NODE=$(nvidia-smi -L | wc -l); fi
fi
(( NPROC_PER_NODE >= 2 )) || { echo "ERROR: need >=2 GPUs; with 1 GPU FSDP does no communication to measure." >&2; exit 1; }
MASTER_PORT="${MASTER_PORT:-50012}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/fsdp_overlap}"
[[ "$OUTPUT_ROOT" = /* ]] || OUTPUT_ROOT="$REPO_ROOT/$OUTPUT_ROOT"

PROFILE_FREQ=$(( PROFILE_WAIT + PROFILE_WARMUP + PROFILE_ACTIVE ))
MAX_ITER=$(( PROFILE_FREQ + 1 ))   # +1 so the trace handler has fired before exit

TARGET_RANKS="[$(seq -s, 0 $(( NPROC_PER_NODE - 1 )))]"
OVERRIDES=(
  "trainer.max_iter=$MAX_ITER"
  "trainer.logging_iter=1"
  "trainer.profiling.enable_profiling=true"
  "trainer.profiling.profile_freq=$PROFILE_FREQ"
  "trainer.profiling.profile_warmup=$PROFILE_WARMUP"
  "trainer.profiling.profile_active=$PROFILE_ACTIVE"
  "trainer.profiling.target_ranks=$TARGET_RANKS"
  "checkpoint.save_iter=1000000000"
)

case "$RECIPE" in
  vision_edge)
    TOML_FILE="examples/toml/sft_config/vision_sft_edge.toml"
    : "${DATASET_PATH:=examples/data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge}"
    : "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"
    : "${WAN_VAE_PATH:=examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
    for v in DATASET_PATH BASE_CHECKPOINT_PATH WAN_VAE_PATH; do
      [[ "${!v}" = /* ]] || printf -v "$v" '%s' "$REPO_ROOT/${!v}"
    done
    [[ -f "$DATASET_PATH/train/video_dataset_file.jsonl" ]] || { echo "ERROR: missing $DATASET_PATH/train/video_dataset_file.jsonl" >&2; exit 1; }
    [[ -d "$BASE_CHECKPOINT_PATH" ]] || { echo "ERROR: BASE_CHECKPOINT_PATH not found: $BASE_CHECKPOINT_PATH (docs/training.md Step 2)" >&2; exit 1; }
    [[ -f "$WAN_VAE_PATH" ]] || { echo "ERROR: WAN_VAE_PATH not found: $WAN_VAE_PATH" >&2; exit 1; }
    export DATASET_PATH BASE_CHECKPOINT_PATH WAN_VAE_PATH
    ;;
  videophy2_edge)
    TOML_FILE="examples/toml/sft_config/videophy2_sft_edge.toml"
    : "${VIDEOPHYSICS_ROOT:?VIDEOPHYSICS_ROOT must be set for videophy2_edge}"
    export VIDEOPHYSICS_ROOT
    [[ -n "${VLM_SAFETENSORS_PATH:-}" ]] && OVERRIDES+=("model.config.policy.backbone.safetensors_path=$VLM_SAFETENSORS_PATH")
    ;;
  *) echo "ERROR: unknown RECIPE=$RECIPE (vision_edge | videophy2_edge)" >&2; exit 1 ;;
esac

# shellcheck disable=SC2206
[[ -n "${EXTRA_OVERRIDES:-}" ]] && OVERRIDES+=( $EXTRA_OVERRIDES )

RUN_TAG="${RECIPE}_${NPROC_PER_NODE}gpu_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUTPUT_ROOT/$RUN_TAG"
mkdir -p "$RUN_DIR"
MARKER="$RUN_DIR/.start_marker"; touch "$MARKER"

# ---- record the hardware/software context (needed to compare across machines) ----
{
  echo "run_tag: $RUN_TAG"; echo "date: $(date -Is)"; echo "host: $(hostname)"
  echo "git: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')"
  echo "recipe: $RECIPE  toml: $TOML_FILE  nproc: $NPROC_PER_NODE"
  echo "schedule: wait=$PROFILE_WAIT warmup=$PROFILE_WARMUP active=$PROFILE_ACTIVE max_iter=$MAX_ITER"
  echo "overrides: ${OVERRIDES[*]}"
  echo "--- nvidia-smi ---"; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  echo "--- topology ---"; nvidia-smi topo -m || true
  echo "--- torch ---"; python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'nccl',torch.cuda.nccl.version())" || true
} > "$RUN_DIR/run_info.txt" 2>&1
cat "$RUN_DIR/run_info.txt"

# NCCL_DEBUG=INFO (INIT subsystem only) records which transport NCCL picked
# (P2P/NVLink vs SHM vs NET) — the single biggest hardware factor here.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}" NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT}"

echo ">>> launching training ($MAX_ITER iters) — log: $RUN_DIR/train.log"
set +e
IMAGINAIRE_OUTPUT_ROOT="$RUN_DIR/imaginaire" PYTHONPATH=. \
  torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
  -m cosmos_framework.scripts.train --sft-toml="$TOML_FILE" -- "${OVERRIDES[@]}" \
  2>&1 | tee "$RUN_DIR/train.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e
grep -E "NCCL INFO (Channel|Connected|.*via )" "$RUN_DIR/train.log" | head -20 > "$RUN_DIR/nccl_transport.txt" || true

TRACE_DIR="$(find "$RUN_DIR" -type f -name 'rank*_trace.json*' -newer "$MARKER" -printf '%h\n' 2>/dev/null | sort -u | tail -1)"
if [[ -z "$TRACE_DIR" ]]; then
  echo "ERROR: no traces found under $RUN_DIR (train exit $TRAIN_EXIT). Check train.log — OOM or the run resumed past max_iter." >&2
  exit 1
fi
[[ $TRAIN_EXIT -ne 0 ]] && echo "WARNING: training exited $TRAIN_EXIT after traces were written (often the final checkpoint save). Continuing."
echo ">>> traces: $TRACE_DIR"; ls -lh "$TRACE_DIR"

# The trainer always writes a final checkpoint at exit; it is useless here and large.
if [[ "${KEEP_CHECKPOINTS:-0}" != 1 ]]; then
  find "$RUN_DIR/imaginaire" -type d -name checkpoints -prune -exec rm -rf {} + 2>/dev/null || true
fi

if [[ "${SKIP_ANALYSIS:-0}" != 1 ]]; then
  python "$REPO_ROOT/tools/fsdp_overlap/analyze_comm_overlap.py" \
    --trace-dir "$TRACE_DIR" --out-dir "$RUN_DIR/analysis" --skip-steps 0
fi
echo ">>> done: $RUN_DIR"
