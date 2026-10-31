export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
#model_name=Linear


python -u run.py \
  --is_training 1 \
  --root_path ../dataset_generation/output_test/2026-10-29/var128_sqrt/ \
  --data_path rep000_lambda1.00_persistence0.95_gamma1.00.npz \
  --model_id synthetic_96_96 \
  --model $model_name \
  --data synthetic \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --itr 1 \
  --train_epochs 1
