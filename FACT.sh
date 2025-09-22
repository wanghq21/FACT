export CUDA_VISIBLE_DEVICES=0
model_name=FACT

e_layers=1
d_model=512
d_ff=512
for len in   96 192 336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/Solar \
    --data_path solar_AL.txt \
    --model_id Solar_96_96 \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --core 0.5 \
    --dropout 0.1 \
    --dilation 1 2 1  \
    --num_kernels 4 \
    --mode 'freq' \
    --padding_mode 'zeros' \
    --learning_rate 0.0005 \
    --d_model $d_model \
    --d_ff $d_ff \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0
done


e_layers=1
d_model=1024
d_ff=1024
for len in       96 192 336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/traffic \
    --data_path traffic.csv \
    --model_id traffic_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --freq h \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --core 0.5 \
    --dropout 0.1 \
    --dilation 1 2 3 2 1 \
    --num_kernels 4 \
    --mode 'freq' \
    --padding_mode 'zeros' \
    --d_model $d_model \
    --d_ff $d_ff \
    --learning_rate 0.001 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --lradj type1 \
    --train_epochs 15 \
    --patience 5 \
    --batch_size 32 \
    --itr 1 \
    --use_norm 1
done


e_layers=1
d_model=512
d_ff=2048
for len in     96 192
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity \
    --data_path electricity.csv \
    --model_id ECL_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --freq h \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.5 \
    --core 0.1 \
    --dilation 1 2 3 4 3  2  1     \
    --num_kernels 4 \
    --mode 'freq' \
    --padding_mode 'zeros' \
    --learning_rate 0.001  \
    --d_model $d_model \
    --d_ff $d_ff \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --train_epochs 15 \
    --batch_size 32 \
    --itr 1 \
    --use_norm 1
done
for len in    336 720
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity \
    --data_path electricity.csv \
    --model_id ECL_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --freq h \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.5 \
    --core 0.3 \
    --dilation 1 2 3 4 3  2    1     \
    --num_kernels 4 \
    --mode 'freq' \
    --padding_mode 'zeros' \
    --learning_rate 0.001  \
    --d_model $d_model \
    --d_ff $d_ff \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --train_epochs 15 \
    --batch_size 32 \
    --itr 1 \
    --use_norm 1
done



e_layers=1
d_model=512
d_ff=2048
for len in     12 24 48 96
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ECL \
    --data_path electricity.npy \
    --model_id ECL2_96_96 \
    --model $model_name \
    --data npy \
    --features M \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.5 \
    --core 0.0 \
    --dilation 1 2 3 2 1  \
    --num_kernels 4 \
    --mode 'freq' \
    --padding_mode 'zeros' \
    --learning_rate 0.001 \
    --d_model $d_model \
    --d_ff $d_ff \
    --enc_in 370 \
    --dec_in 370 \
    --c_out 370 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 1
done


e_layers=1
d_model=1024
d_ff=1024
for len in   12 24 48 96 
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ECL \
    --data_path traffic.npy \
    --model_id traffic2_96_96 \
    --model $model_name \
    --data npy \
    --features M \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --dropout 0.1 \
    --core 0.3 \
    --dilation 1 2 3 2 1  \
    --num_kernels 4 \
    --mode 'freq' \
    --padding_mode 'zeros' \
    --learning_rate 0.001 \
    --d_model $d_model \
    --d_ff $d_ff \
    --enc_in 963 \
    --dec_in 963 \
    --c_out 963 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 1
done




e_layers=1
d_model=512
d_ff=512
for len in    12 24 48 96
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS \
    --data_path PEMSD7.npy \
    --model_id PEMSD7_96_96 \
    --model $model_name \
    --data npy \
    --features M \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --core 1.0 \
    --dropout 0.7 \
    --dilation 1 2 3 2  1 \
    --num_kernels 4 \
    --mode 'freq' \
    --padding_mode 'zeros' \
    --learning_rate 0.001 \
    --d_model $d_model \
    --d_ff $d_ff \
    --enc_in 228 \
    --dec_in 228 \
    --c_out 228 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0
done

e_layers=1
d_model=512
d_ff=2048
for len in     12 24 48 96
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS \
    --data_path PEMS-BAY.csv \
    --model_id PEMS-BAY_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --freq t \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --core 0.1 \
    --dropout 0.7 \
    --dilation 1 2 3 4 3  2 1  \
    --num_kernels 4 \
    --mode 'freq' \
    --padding_mode 'zeros' \
    --d_model $d_model \
    --d_ff $d_ff \
    --learning_rate 0.001 \
    --enc_in 325 \
    --dec_in 325 \
    --c_out 325 \
    --des 'Exp' \
    --lradj type1 \
    --patience 5 \
    --batch_size 32 \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0
done


e_layers=1
d_model=512
d_ff=512
for len in    12 24 48 96
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/METR \
    --data_path METR-LA.csv \
    --model_id METR-LA_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --freq t \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --core 0.9 \
    --dropout 0.7 \
    --dilation 1 2  1   \
    --num_kernels 4 \
    --mode 'freq' \
    --padding_mode 'zeros' \
    --d_model $d_model \
    --d_ff $d_ff \
    --learning_rate 0.001 \
    --enc_in 207 \
    --dec_in 207 \
    --c_out 207 \
    --des 'Exp' \
    --lradj type1 \
    --patience 3 \
    --batch_size 32 \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0
done


e_layers=1
d_model=512
d_ff=2048
for len in    12 24 48 96
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS03.npz \
    --model_id PEMS03_96_12 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --enc_in 358 \
    --dec_in 358 \
    --c_out 358 \
    --core 0.3 \
    --dilation 1 2 3 2 1       \
    --num_kernels 4 \
    --mode 'freq' \
    --padding_mode 'zeros' \
    --des 'Exp' \
    --d_model $d_model \
    --d_ff $d_ff \
    --learning_rate 0.001 \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0
done




e_layers=1
d_model=512
d_ff=2048
for len in    12 24 48 96
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id PEMS04_96_12 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --core 0.3 \
    --dilation 1 2 3 2  1     \
    --num_kernels 4 \
    --mode 'freq' \
    --padding_mode 'zeros' \
    --des 'Exp' \
    --d_model $d_model \
    --d_ff $d_ff \
    --learning_rate 0.001 \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0
done


e_layers=1
d_model=512
d_ff=2048
for len in   12 24 48 96
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07_96_96 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883 \
    --core 0.3 \
    --dilation 1 2 3 4 3  2 1 \
    --num_kernels 4 \
    --mode 'freq' \
    --padding_mode 'zeros' \
    --des 'Exp' \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size 32 \
    --learning_rate 0.001 \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0
done


e_layers=1
d_model=512
d_ff=2048
for len in    12 24 48 96
do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS08.npz \
    --model_id PEMS08_96_96 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len $len \
    --e_layers $e_layers \
    --enc_in 170 \
    --dec_in 170 \
    --c_out 170 \
    --core 0.0 \
    --dropout 0.1 \
    --dilation 1 2 3 2  1  \
    --num_kernels 4 \
    --mode 'freq' \
    --padding_mode 'zeros' \
    --des 'Exp' \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size 32 \
    --learning_rate 0.001 \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0
done