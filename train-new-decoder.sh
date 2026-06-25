python train.py \
  --task cy_polytope \
  --dump_path exp/cy_points_1M \
  --exp_name cy_points_1M \
  \
  --architecture encoder_decoder \
  --enc_emb_dim 256 --dec_emb_dim 256 \
  --n_enc_layers 4  --n_dec_layers 1 \
  --n_enc_heads 8   --n_dec_heads 8 \
  --norm layernorm  --activation gelu \
  --enc_pos_emb abs_learned --dec_pos_emb abs_learned \
  --share_inout_emb true \
  --max_len 320 --max_output_len 512 \
  --base 100 \
  \
  --reload_checkpoint exp/cy_facets_10M/best-valid_CY_POLYTOPE_greedy_acc.pth \
  --freeze_encoder true \
  \
  --reload_data cy_polytope:../data/axolver-1M-point_count/train.data \
  --eval_data ../data/axolver-1M-point_count/valid.data,../data/axolver-1M-point_count/test.data \
  --eval_size 500 \
  \
  --optimizer adam,lr=0.0001 \
  --batch_size 32 \
  --epoch_size 30000 \
  --amp true
