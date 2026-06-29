#!/bin/sh
# Usage: sh train-encoder.sh task=<name> [encoder-checkpoint=<task|path>]

SIZE="10M"
TASK=""
ENC_CHECKPOINT_ARG=""

for arg in "$@"; do
    case "$arg" in
        task=*)               TASK="${arg#task=}" ;;
        encoder-checkpoint=*) ENC_CHECKPOINT_ARG="${arg#encoder-checkpoint=}" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if [ -z "$TASK" ]; then
    echo "Usage: $0 task=<name> [encoder-checkpoint=<task|path>]" >&2
    exit 1
fi

DUMP_PATH="exp/CY-4d/encoder-${TASK}-${SIZE}"
EXP_NAME="decoder-${TASK}"
RELOAD_DATA="cy_polytope:../data/axolver-${TASK}-${SIZE}/train.data"
EVAL_DATA="../data/axolver-${TASK}-${SIZE}/valid.data,../data/axolver-${TASK}-${SIZE}/test.data"

# Resolve checkpoint: explicit task/path, auto-resume from existing, or null to start fresh
ENC_CHECKPOINT=""
DEC_CHECKPOINT=""
case "$ENC_CHECKPOINT_ARG" in
    null) ;;
    */*|*.pth) ENC_CHECKPOINT="$ENC_CHECKPOINT_ARG" ;;
    ?*)
        DIR=$(ls -d "exp/CY-4d/encoder-${ENC_CHECKPOINT_ARG}-${SIZE}/decoder-${ENC_CHECKPOINT_ARG}"/[0-9]*/ 2>/dev/null | sort -n | tail -1)
        [ -z "$DIR" ] && echo "ERROR: No checkpoint found for task ${ENC_CHECKPOINT_ARG}" >&2 && exit 1
        ENC_CHECKPOINT="${DIR}checkpoint-encoder.pth"
        DEC_CHECKPOINT="${DIR}checkpoint-decoder-cy_polytope.pth"
        ;;
    *)  EXISTING=$(ls -d "${DUMP_PATH}/${EXP_NAME}"/[0-9]*/ 2>/dev/null | sort -n | tail -1)
        [ -n "$EXISTING" ] && ENC_CHECKPOINT="${EXISTING}checkpoint-encoder.pth" \
                           && DEC_CHECKPOINT="${EXISTING}checkpoint-decoder-cy_polytope.pth" ;;
esac

echo ""
echo "=== Encoder Training ==="
echo "  task:         cy_polytope (${TASK})	# task type and data variant"
echo "  dump_path:    ${DUMP_PATH}	# where checkpoints and logs are saved"
echo "  exp_name:     ${EXP_NAME}	# experiment subfolder name"
echo "  reload_data:  ${RELOAD_DATA}	# training data"
echo "  eval_data:    ${EVAL_DATA}	# validation and test data"
echo "  n_dec_layers: 1	# decoder transformer layers (1 = lightweight head)"
echo "  max_len:      320	# max input tokens (overrides default 256)"
echo "  base:         1000	# base for integer tokenization (overrides default 100)"
echo "  eval_size:    5000	# examples per eval split (overrides default 10000)"
echo "  amp:          true	# automatic mixed precision (~2x faster)"
if [ -n "$ENC_CHECKPOINT" ]; then
    EPOCH=$(python3 tools/read_epoch.py "${ENC_CHECKPOINT}" 2>/dev/null)
    echo "  checkpoint:   ${ENC_CHECKPOINT} (epoch ${EPOCH})"
fi
[ -n "$SLURM_JOB_ID" ] && echo "  slurm_job_id: ${SLURM_JOB_ID}"
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
[ -n "$ENC_CHECKPOINT" ] && EXTRA_ARGS="$EXTRA_ARGS --reload_encoder_checkpoint ${ENC_CHECKPOINT}"
[ -n "$DEC_CHECKPOINT" ] && EXTRA_ARGS="$EXTRA_ARGS --reload_decoder_checkpoint ${DEC_CHECKPOINT}"

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
  --eval_size 5000 \
  --amp true \
  $EXTRA_ARGS
