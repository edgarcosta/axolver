#!/bin/sh
# Usage: sh eval-decoder.sh enc-task=<name> dec-task=<name> [frozen=true] [size=10M]

SIZE="10M"
ENC_TASK=""
DEC_TASK=""
FROZEN="false"

for arg in "$@"; do
    case "$arg" in
        enc-task=*) ENC_TASK="${arg#enc-task=}" ;;
        dec-task=*) DEC_TASK="${arg#dec-task=}" ;;
        frozen=*)   FROZEN="${arg#frozen=}" ;;
        size=*)     SIZE="${arg#size=}" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if [ -z "$ENC_TASK" ] || [ -z "$DEC_TASK" ]; then
    echo "Usage: $0 enc-task=<name> dec-task=<name> [frozen=true] [size=10M]" >&2
    exit 1
fi

[ "$FROZEN" = "true" ] && TAG="-frozen" || TAG=""
DEC_EXP_NAME="decoder-${DEC_TASK}${TAG}"
DEC_EXP="exp/CY-4d/encoder-${ENC_TASK}-${SIZE}/${DEC_EXP_NAME}"

# Pick the latest job-ID subdirectory
DEC_DIR=$(ls -d "${DEC_EXP}"/[0-9]*/ 2>/dev/null | sort -n | tail -1)
if [ -z "$DEC_DIR" ]; then
    echo "ERROR: No checkpoint found in ${DEC_EXP}" >&2; exit 1
fi

ENC_EXP="exp/CY-4d/encoder-${ENC_TASK}-${SIZE}/decoder-${ENC_TASK}"
ENC_DIR=$(ls -d "${ENC_EXP}"/[0-9]*/ 2>/dev/null | sort -n | tail -1)
if [ -z "$ENC_DIR" ]; then
    echo "ERROR: No checkpoint found in ${ENC_EXP}" >&2; exit 1
fi

# Hyphenated dec-task = multiple heads (h11-h12 -> "h11,h12"); first = primary.
DEC_HEADS=$(echo "$DEC_TASK" | tr '-' ',')
DEC_PRIMARY="${DEC_TASK%%-*}"

ENC_CHECKPOINT="${ENC_DIR}checkpoint-encoder.pth"
DEC_CHECKPOINT="${DEC_DIR}checkpoint-decoder-${DEC_PRIMARY}.pth"

DUMP_PATH="exp/CY-4d/encoder-${ENC_TASK}-${SIZE}"
EXP_NAME="eval-dec-${DEC_TASK}${TAG}"
RELOAD_DATA="cy_polytope,${DEC_HEADS}:../data/axolver-${DEC_TASK}-${SIZE}/test.data"
EVAL_DATA="cy_polytope,${DEC_HEADS}:../data/axolver-${DEC_TASK}-${SIZE}/test.data"

echo ""
echo "=== Decoder Evaluation ==="
echo "  enc-task:      ${ENC_TASK}	# encoder experiment to load"
echo "  dec-task:      ${DEC_TASK}${TAG}	# decoder experiment to load"
[ "$FROZEN" = "true" ] && echo "  freeze_encoder: true	# encoder weights are fixed"
echo "  dump_path:     ${DUMP_PATH}	# where eval results are saved"
echo "  exp_name:      ${EXP_NAME}	# experiment subfolder name"
echo "  reload_data:   ${RELOAD_DATA}	# satisfies DataLoader init (not used for training)"
echo "  eval_data:     ${EVAL_DATA}	# test data"
echo "  n_dec_layers:  1	# decoder transformer layers (1 = lightweight head)"
echo "  max_len:       320	# max input tokens (overrides default 256)"
echo "  base:          1000	# base for integer tokenization (overrides default 100)"
echo "  eval_size:     5000	# examples evaluated (overrides default 10000)"
echo "  encoder:       ${ENC_CHECKPOINT}"
echo "  decoder:       ${DEC_CHECKPOINT}"
echo ""
printf "Proceed? [y/N] "
read CONFIRM
case "$CONFIRM" in
    y|Y) ;;
    *) echo "Aborted."; exit 0 ;;
esac

if command -v module >/dev/null 2>&1; then
    module load cuda/13
fi
source ~/venv_axolver/bin/activate

EXTRA_ARGS=""
[ "$FROZEN" = "true" ] && EXTRA_ARGS="$EXTRA_ARGS --freeze_encoder true"
python3 -c "import torch; exit(0 if torch.cuda.is_available() or torch.xpu.is_available() or torch.backends.mps.is_available() else 1)" 2>/dev/null \
    || EXTRA_ARGS="$EXTRA_ARGS --cpu true"

# shellcheck disable=SC2086
python -u train.py \
  --task cy_polytope \
  --dump_path "${DUMP_PATH}" \
  --exp_name "${EXP_NAME}" \
  --n_dec_layers 1 \
  --max_len 320 \
  --base 1000 \
  --reload_data "${RELOAD_DATA}" \
  --reload_encoder_checkpoint "${ENC_CHECKPOINT}" \
  --reload_decoder_checkpoint "${DEC_CHECKPOINT}" \
  --eval_data "${EVAL_DATA}" \
  --eval_only true \
  --eval_size 5000 \
  $EXTRA_ARGS
