# Lấy đường dẫn gốc
model_name=CA_MLP
num_wavelets=0.5
kernel_size=3

# Tạo thư mục logs nếu chưa có
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# Mảng các giá trị num_blocks và pred_len
blocks_list="1 2 3 4 5"
pred_lens="96 192 336 720"

echo "======= BẮT ĐẦU THỬ NGHIỆM NUM_BLOCKS ======="

for num_blocks in $blocks_list
do
    echo "--------------------------------------------------"
    echo "Đang chạy với num_blocks = $num_blocks (e_layers=$num_blocks, d_layers=$num_blocks)"
    
    sum_mse=0
    sum_mae=0
    count=0

    for pred_len in $pred_lens
    do
        echo "  -> Đang chạy pred_len = $pred_len..."
        
        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --model_id ETTh2_512_$pred_len \
          --model $model_name \
          --data ETTh2 \
          --root_path ./dataset/ETT-small/ \
          --data_path ETTh2.csv \
          --features M \
          --target OT \
          --freq h \
          --seq_len 512 \
          --label_len 0 \
          --pred_len $pred_len \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --d_model 16 \
          --n_heads 4 \
          --e_layers $num_blocks \
          --d_layers $num_blocks \
          --d_ff 32 \
          --factor 1 \
          --dropout 0.1 \
          --channel_independence 1 \
          --batch_size 32 \
          --learning_rate 0.001 \
          --train_epochs 100 \
          --patience 10 \
          --lradj 'cosine' \
          --pct_start 0.2 \
          --num_wavelets $num_wavelets \
          --kernel_size $kernel_size \
          --des Exp_CA_MLP_researching \
          --itr 1 | tee current_run.log

        # Trích xuất MSE và MAE từ file log vừa ghi
        mse=$(grep "mse:" current_run.log | tail -n 1 | sed -n 's/.*mse:\([^,]*\).*/\1/p')
        mae=$(grep "mae:" current_run.log | tail -n 1 | sed -n 's/.*mae:\([^,]*\).*/\1/p')

        if [ -z "$mse" ]; then
            mse=$(grep "mse" current_run.log | tail -n 1 | awk -F'mse:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
            mae=$(grep "mae" current_run.log | tail -n 1 | awk -F'mae:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
        fi

        if [ -n "$mse" ] && [ -n "$mae" ]; then
            echo "     [KQ] pred_len $pred_len: MSE=$mse, MAE=$mae"
            sum_mse=$(echo "$sum_mse + $mse" | bc -l)
            sum_mae=$(echo "$sum_mae + $mae" | bc -l)
            count=$((count + 1))
        else
            echo "     [!] LỖI: Không lấy được kết quả từ log cho pred_len $pred_len"
        fi
        
        rm current_run.log
    done

    # Tính và in trung bình cho mỗi num_blocks
    if [ $count -gt 0 ]; then
        avg_mse=$(echo "$sum_mse / $count" | bc -l)
        avg_mae=$(echo "$sum_mae / $count" | bc -l)
        echo ""
        echo ">>> KẾT QUẢ TRUNG BÌNH CHO num_blocks $num_blocks:"
        echo "    MSE TB: $avg_mse"
        echo "    MAE TB: $avg_mae"
        echo ""
    fi
done

echo "======= HOÀN THÀNH TẤT CẢ ======="