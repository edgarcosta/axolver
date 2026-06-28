#!/bin/sh

module load cuda/13
source ~/venv_axolver/bin/activate

TAG="-h11-h12"
TASK="h12"
SIZE="10M"

python train.py \
  --task cy_polytope \
  --dump_path exp/cy_${TASK}_${SIZE} \
  --exp_name cy_${TASK}_${SIZE}${TAG} \
  \
  --architecture encoder_decoder \
  --enc_emb_dim 256 --dec_emb_dim 256 \
  --n_enc_layers 4  --n_dec_layers 1 \
  --n_enc_heads 8   --n_dec_heads 8 \
  --norm layernorm  --activation gelu \
  --enc_pos_emb abs_learned --dec_pos_emb abs_learned \
  --share_inout_emb true \
  --max_len 320 --max_output_len 512 \
  --base 1000 \
  \
  --freeze_encoder false \
  --reload_checkpoint exp/cy_h11-h12_10M/cy_h11-h12_10M-scratch/16803238/checkpoint-encoder.pth \
  \
  --reload_data cy_polytope:../data/axolver-${TASK}-${SIZE}/train.data \
  --eval_data ../data/axolver-${TASK}-${SIZE}/valid.data,../data/axolver-${TASK}-${SIZE}/test.data \
  --eval_size 5000 \
  \
  --batch_size 32 \
  --amp true
#  --optimizer adam,lr=0.0001 \
#  --epoch_size 30000 \
#  --reload_checkpoint exp/cy_facets_10M-original/1/best-valid_CY_POLYTOPE_greedy_acc.pth \
#  --reload_checkpoint exp/cy_h11-h12_10M/cy_h11-h12_10M/16802767/checkpoint-encoder.pth \
