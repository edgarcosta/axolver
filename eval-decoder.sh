#!/bin/sh
# Usage: sh eval-decoder.sh enc-task=<name> dec-task=<name> [size=10M]

SIZE="10M"
ENC_TASK=""
DEC_TASK=""

for arg in "$@"; do
    case "$arg" in
        enc-task=*) ENC_TASK="${arg#enc-task=}" ;;
        dec-task=*) DEC_TASK="${arg#dec-task=}" ;;
        size=*)     SIZE="${arg#size=}" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if [ -z "$ENC_TASK" ] || [ -z "$DEC_TASK" ]; then
    echo "Usage: $0 enc-task=<name> dec-task=<name> [size=10M]" >&2
    exit 1
fi

ENC_EXP="exp/CY-4d/encoder-${ENC_TASK}-${SIZE}/decoder-${ENC_TASK}"
DEC_EXP="exp/CY-4d/encoder-${ENC_TASK}-${SIZE}/decoder-${DEC_TASK}"

# Pick the latest job-ID subdirectory
ENC_DIR=$(ls -d "${ENC_EXP}"/[0-9]*/ 2>/dev/null | sort -n | tail -1)
DEC_DIR=$(ls -d "${DEC_EXP}"/[0-9]*/ 2>/dev/null | sort -n | tail -1)

if [ -z "$ENC_DIR" ]; then
    echo "ERROR: No checkpoint found in ${ENC_EXP}" >&2; exit 1
fi
if [ -z "$DEC_DIR" ]; then
    echo "ERROR: No checkpoint found in ${DEC_EXP}" >&2; exit 1
fi

ENC_CHECKPOINT="${ENC_DIR}checkpoint-encoder.pth"
DEC_CHECKPOINT="${DEC_DIR}checkpoint-decoder-cy_polytope.pth"

DUMP_PATH="exp/CY-4d/encoder-${ENC_TASK}-${SIZE}"
EXP_NAME="eval-dec-${DEC_TASK}"
RELOAD_DATA="cy_polytope:../data/axolver-${DEC_TASK}-${SIZE}/test.data"
EVAL_DATA="../data/axolver-${DEC_TASK}-${SIZE}/test.data"

echo ""
echo "=== Decoder Evaluation ==="
echo "  enc-task:      ${ENC_TASK}	# encoder experiment to load"
echo "  dec-task:      ${DEC_TASK}	# decoder experiment to load"
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

module load cuda/13
source ~/venv_axolver/bin/activate

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
  --eval_size 5000
