export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
#model_name=Linear

python -u run.py \
  --is_training 1 \
  --root_path ../dataset_generation/ \
  --data_path beijing_panel_segments.npz \
  --model_id beijing_itransformer_96_96 \
  --target S1_SO2 \
  --model $model_name \
  --data beijing \
  --features M \
  --freq h \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 108 \
  --dec_in 108 \
  --c_out 108 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --itr 1

