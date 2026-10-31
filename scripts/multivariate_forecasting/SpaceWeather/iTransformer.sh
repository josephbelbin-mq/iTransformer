export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
#model_name=Linear

python -u run.py \
  --is_training 1 \
  --root_path ../dataset_generation/ \
  --data_path space_weather_clean.csv \
  --model_id space_weather_96_96 \
  --target sunspot_number \
  --model $model_name \
  --data custom \
  --features M \
  --freq d \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 30 \
  --dec_in 30 \
  --c_out 30 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.005 \
  --itr 1

