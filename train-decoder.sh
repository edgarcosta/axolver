#!/bin/sh
# Usage: sh train-decoder.sh task=<name> [frozen=true] [encoder-checkpoint=<task|path>] [decoder-checkpoint=<task|path|null>]

SIZE="10M"
TASK=""
FROZEN="false"
ENC_CHECKPOINT_ARG=""
DEC_CHECKPOINT_ARG=""

for arg in "$@"; do
    case "$arg" in
        task=*)               TASK="${arg#task=}" ;;
        frozen=*)             FROZEN="${arg#frozen=}" ;;
        encoder-checkpoint=*) ENC_CHECKPOINT_ARG="${arg#encoder-checkpoint=}" ;;
        decoder-checkpoint=*) DEC_CHECKPOINT_ARG="${arg#decoder-checkpoint=}" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if [ -z "$TASK" ]; then
    echo "Usage: $0 task=<name> [frozen=true] [encoder-checkpoint=<task|path>] [decoder-checkpoint=<task|path|null>]" >&2
    exit 1
fi

# Hyphenated task = multiple decoder heads (h11-h12 -> "h11,h12"); first = primary.
HEADS=$(echo "$TASK" | tr '-' ',')
PRIMARY="${TASK%%-*}"
DATADIR="../data/axolver-${TASK}-${SIZE}"

resolve_checkpoint() {
    ARG="$1"; SUFFIX="$2"; ENC_TASK="$3"
    case "$ARG" in
        */*|*.pth) echo "$ARG" ;;
        *) DIR=$(ls -d "exp/CY-4d/encoder-${ENC_TASK:-$ARG}-${SIZE}/decoder-${ARG}"/[0-9]*/ 2>/dev/null | sort -n | tail -1)
           [ -z "$DIR" ] && echo "ERROR: No checkpoint found for task ${ARG}" >&2 && exit 1
           echo "${DIR}${SUFFIX}" ;;
    esac
}

# Resolve encoder checkpoint
ENC_CHECKPOINT=""
ENC_TASK_NAME=""
if [ -n "$ENC_CHECKPOINT_ARG" ]; then
    case "$ENC_CHECKPOINT_ARG" in
        */*|*.pth) ENC_CHECKPOINT="$ENC_CHECKPOINT_ARG" ;;
        *) ENC_TASK_NAME="$ENC_CHECKPOINT_ARG"
           ENC_CHECKPOINT=$(resolve_checkpoint "$ENC_TASK_NAME" "checkpoint-encoder.pth" "$ENC_TASK_NAME") || exit 1 ;;
    esac
fi

[ "$FROZEN" = "true" ] && TAG="-frozen" || TAG=""

DUMP_PATH="exp/CY-4d${ENC_TASK_NAME:+/encoder-${ENC_TASK_NAME}-${SIZE}}"
EXP_NAME="decoder-${TASK}${TAG}"
RELOAD_DATA="cy_polytope,${HEADS}:${DATADIR}/train.data"
EVAL_DATA="cy_polytope,${HEADS}:${DATADIR}/valid.data"
TEST_DATA="cy_polytope,${HEADS}:${DATADIR}/test.data"

# Resolve explicit decoder checkpoint (named by the donor task's primary head)
DEC_CHECKPOINT=""
if [ -n "$DEC_CHECKPOINT_ARG" ]; then
    DEC_CHECKPOINT=$(resolve_checkpoint "$DEC_CHECKPOINT_ARG" "checkpoint-decoder-${DEC_CHECKPOINT_ARG%%-*}.pth" "$ENC_TASK_NAME") || exit 1
fi

# Auto-resume from existing run unless explicit checkpoint given or null requested.
# (A SLURM requeue resumes in-place via the trainer, restoring all head decoders;
# this cross-dir fallback warm-starts the encoder + primary decoder only.)
EXISTING=$(ls -d "${DUMP_PATH}/${EXP_NAME}"/[0-9]*/ 2>/dev/null | sort -n | tail -1)
if [ -z "$DEC_CHECKPOINT" ] && [ "$DEC_CHECKPOINT_ARG" != "null" ] && [ -n "$EXISTING" ]; then
    # Don't override an explicitly-given encoder checkpoint (e.g. encoder-checkpoint=h11-h12)
    [ -z "$ENC_CHECKPOINT_ARG" ] && ENC_CHECKPOINT="${EXISTING}checkpoint-encoder.pth"
    DEC_CHECKPOINT="${EXISTING}checkpoint-decoder-${PRIMARY}.pth"
fi

echo ""
echo "=== Decoder Training ==="
echo "  task:          cy_polytope (${TASK})	# task type and data variant"
echo "  dump_path:     ${DUMP_PATH}	# where checkpoints and logs are saved"
echo "  exp_name:      ${EXP_NAME}	# experiment subfolder name"
echo "  heads:         ${HEADS} (primary ${PRIMARY})	# decoder heads from the task name"
echo "  reload_data:   ${RELOAD_DATA}	# training data (column spec)"
echo "  eval_data:     ${EVAL_DATA}	# validation data"
echo "  test_data:     ${TEST_DATA}	# test data"
echo "  n_dec_layers:  1	# decoder transformer layers (1 = lightweight head)"
echo "  max_len:       320	# max input tokens (overrides default 256)"
echo "  base:          1000	# base for integer tokenization (overrides default 100)"
echo "  eval_size:     5000	# examples per eval split (overrides default 10000)"
echo "  amp:           true	# automatic mixed precision (~2x faster)"
[ "$FROZEN" = "true" ] && echo "  freeze_encoder: true	# encoder weights are fixed, only decoder trains"
if [ -n "$ENC_CHECKPOINT" ]; then
    EPOCH=$(python3 tools/read_epoch.py "${ENC_CHECKPOINT}" 2>/dev/null)
    echo "  enc-checkpoint: ${ENC_CHECKPOINT} (epoch ${EPOCH})"
fi
if [ "$DEC_CHECKPOINT_ARG" = "null" ]; then
    echo "  dec-checkpoint: (none — decoder starts from random init)"
elif [ -n "$DEC_CHECKPOINT" ]; then
    EPOCH=$(python3 tools/read_epoch.py "${DEC_CHECKPOINT}" 2>/dev/null)
    echo "  dec-checkpoint: ${DEC_CHECKPOINT} (epoch ${EPOCH})"
fi
[ -n "$SLURM_JOB_ID" ] && echo "  slurm_job_id:  ${SLURM_JOB_ID}"
echo ""
printf "Proceed? [y/N] "
read CONFIRM
case "$CONFIRM" in
    y|Y) ;;
    *) echo "Aborted."; exit 0 ;;
esac

module load cuda/13
source ~/venv_axolver/bin/activate

EXTRA_ARGS=""
[ "$FROZEN" = "true" ]             && EXTRA_ARGS="$EXTRA_ARGS --freeze_encoder true"
[ -n "$ENC_CHECKPOINT" ]           && EXTRA_ARGS="$EXTRA_ARGS --reload_encoder_checkpoint ${ENC_CHECKPOINT}"
[ -n "$DEC_CHECKPOINT" ]           && EXTRA_ARGS="$EXTRA_ARGS --reload_decoder_checkpoint ${DEC_CHECKPOINT}"
[ "$DEC_CHECKPOINT_ARG" = "null" ] && EXTRA_ARGS="$EXTRA_ARGS --ignore_decoder_checkpoint true"

# shellcheck disable=SC2086
python -u train.py \
  --task cy_polytope \
  --dump_path "${DUMP_PATH}" \
  --exp_name "${EXP_NAME}" \
  --n_dec_layers 1 \
  --max_len 320 \
  --base 1000 \
  --reload_data "${RELOAD_DATA}" \
  --eval_data "${EVAL_DATA}" \
  --test_data "${TEST_DATA}" \
  --eval_size 5000 \
  --amp true \
  $EXTRA_ARGS
