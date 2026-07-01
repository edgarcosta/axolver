#!/bin/sh
# Usage: sh eval-decoder.sh dec-task=<name> enc-task=<name> [frozen=true] [size=10M]
#        sh eval-decoder.sh dec-task=<name> enc-checkpoint=<path> dec-checkpoint=<path>

SIZE="10M"
ENC_TASK=""
DEC_TASK=""
FROZEN="false"
ENC_CHECKPOINT_ARG=""
DEC_CHECKPOINT_ARG=""

for arg in "$@"; do
    case "$arg" in
        enc-task=*)       ENC_TASK="${arg#enc-task=}" ;;
        dec-task=*)       DEC_TASK="${arg#dec-task=}" ;;
        frozen=*)         FROZEN="${arg#frozen=}" ;;
        size=*)           SIZE="${arg#size=}" ;;
        enc-checkpoint=*) ENC_CHECKPOINT_ARG="${arg#enc-checkpoint=}" ;;
        dec-checkpoint=*) DEC_CHECKPOINT_ARG="${arg#dec-checkpoint=}" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if [ -z "$DEC_TASK" ]; then
    echo "Usage: $0 dec-task=<name> enc-task=<name> [frozen=true] [size=10M]" >&2
    echo "       $0 dec-task=<name> enc-checkpoint=<path> dec-checkpoint=<path>" >&2
    exit 1
fi

[ "$FROZEN" = "true" ] && TAG="-frozen" || TAG=""

# Hyphenated dec-task = multiple heads (h11-h12 -> "h11,h12"); first = primary.
DEC_HEADS=$(echo "$DEC_TASK" | tr '-' ',')
DEC_PRIMARY="${DEC_TASK%%-*}"

# Resolve encoder checkpoint
if [ -n "$ENC_CHECKPOINT_ARG" ]; then
    ENC_CHECKPOINT="$ENC_CHECKPOINT_ARG"
else
    if [ -z "$ENC_TASK" ]; then
        echo "ERROR: enc-task is required when enc-checkpoint is not given" >&2; exit 1
    fi
    ENC_EXP="exp/CY-4d/encoder-${ENC_TASK}-${SIZE}/decoder-${ENC_TASK}"
    ENC_DIR=$(ls -d "${ENC_EXP}"/[0-9]*/ 2>/dev/null | sort -n | tail -1)
    if [ -z "$ENC_DIR" ]; then
        echo "ERROR: No checkpoint found in ${ENC_EXP}" >&2; exit 1
    fi
    ENC_CHECKPOINT="${ENC_DIR}checkpoint-encoder.pth"
fi

# Resolve decoder checkpoint
if [ -n "$DEC_CHECKPOINT_ARG" ]; then
    DEC_CHECKPOINT="$DEC_CHECKPOINT_ARG"
else
    if [ -z "$ENC_TASK" ]; then
        echo "ERROR: enc-task is required when dec-checkpoint is not given" >&2; exit 1
    fi
    DEC_EXP="exp/CY-4d/encoder-${ENC_TASK}-${SIZE}/decoder-${DEC_TASK}${TAG}"
    DEC_DIR=$(ls -d "${DEC_EXP}"/[0-9]*/ 2>/dev/null | sort -n | tail -1)
    if [ -z "$DEC_DIR" ]; then
        echo "ERROR: No checkpoint found in ${DEC_EXP}" >&2; exit 1
    fi
    DEC_CHECKPOINT="${DEC_DIR}checkpoint-decoder-${DEC_PRIMARY}.pth"
fi

# Derive dump_path: from enc-task if given, else 3 levels up from the encoder checkpoint
# (checkpoint-encoder.pth -> job-dir -> exp-name-dir -> encoder-exp-dir)
if [ -n "$ENC_TASK" ]; then
    DUMP_PATH="exp/CY-4d/encoder-${ENC_TASK}-${SIZE}"
else
    DUMP_PATH=$(dirname "$(dirname "$(dirname "$ENC_CHECKPOINT")")")
fi
EXP_NAME="eval-dec-${DEC_TASK}${TAG}"

RELOAD_DATA="cy_polytope,${DEC_HEADS}:../data/axolver-${DEC_TASK}-${SIZE}/test.data"
EVAL_DATA="cy_polytope,${DEC_HEADS}:../data/axolver-${DEC_TASK}-${SIZE}/test.data"

# Activate venv early so CPU detection can use torch
if command -v module >/dev/null 2>&1; then
    module load cuda/13
fi
source ~/venv_axolver/bin/activate

# Build the command (set -- lets the same $@ be printed and executed)
set -- python -u train.py \
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
  --eval_size 5000
[ "$FROZEN" = "true" ] && set -- "$@" --freeze_encoder true
python3 -c "import torch; exit(0 if torch.cuda.is_available() or torch.xpu.is_available() or torch.backends.mps.is_available() else 1)" 2>/dev/null \
    || set -- "$@" --cpu true

ENC_EPOCH=$(python3 tools/read_epoch.py "${ENC_CHECKPOINT}" 2>/dev/null)
DEC_EPOCH=$(python3 tools/read_epoch.py "${DEC_CHECKPOINT}" 2>/dev/null)

echo ""
echo "=== Decoder Evaluation ==="
echo "  dec-task:      ${DEC_TASK}${TAG}	# decoder experiment"
[ -n "$ENC_TASK" ] && echo "  enc-task:      ${ENC_TASK}	# encoder experiment"
[ "$FROZEN" = "true" ] && echo "  freeze_encoder: true	# encoder weights are fixed"
echo "  dump_path:     ${DUMP_PATH}	# where eval results are saved"
echo "  exp_name:      ${EXP_NAME}	# experiment subfolder name"
echo "  eval_data:     ${EVAL_DATA}	# test data"
echo "  n_dec_layers:  1	# decoder transformer layers (1 = lightweight head)"
echo "  max_len:       320	# max input tokens (overrides default 256)"
echo "  base:          1000	# base for integer tokenization (overrides default 100)"
echo "  eval_size:     5000	# examples evaluated (overrides default 10000)"
echo "  encoder:       ${ENC_CHECKPOINT} (epoch ${ENC_EPOCH})"
echo "  decoder:       ${DEC_CHECKPOINT} (epoch ${DEC_EPOCH})"
echo ""
echo "  command: $*"
echo ""
printf "Proceed? [y/N] "
read CONFIRM
case "$CONFIRM" in
    y|Y) ;;
    *) echo "Aborted."; exit 0 ;;
esac

"$@"
