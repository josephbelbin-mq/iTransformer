export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
#model_name=Linear

python -u run.py \
  --is_training 1 \
  --root_path ../dataset_generation/ \
  --data_path space_weather_reduced.csv \
  --model_id space_weather_96_96 \
  --target sunspot_number \
  --model $model_name \
  --data custom \
  --features M \
  --freq d \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 20 \
  --dec_in 20 \
  --c_out 20 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --patience 10 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --itr 1

