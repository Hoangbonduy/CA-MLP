import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Đường dẫn tới thư mục cha
BASE_DIR = 'CA-MLP/visualization' 

models = [
    'Autoformer', 
    'TimesNet', 
    'PatchTST', 
    'iTransformer', 
    'TimeMixer', 
    'CA_MLP'
]

data_dict = {}
expected_length = None

print("Đang xử lý dữ liệu để loại bỏ nhiễu chồng chéo...")
for model in models:
    pred_path = os.path.join(BASE_DIR, model, 'pred.npy')
    true_path = os.path.join(BASE_DIR, model, 'true.npy')
    
    if os.path.exists(pred_path) and os.path.exists(true_path):
        pred = np.load(pred_path)
        true = np.load(true_path)
        
        # 1. Tính MSE trên toàn bộ cấu trúc dữ liệu nguyên bản (để giữ độ chính xác tuyệt đối)
        if pred.ndim >= 2:
            mse_global = np.mean((true[:, :, -1] - pred[:, :, -1]) ** 2) if pred.ndim == 3 else np.mean((true[:, -1] - pred[:, -1]) ** 2)
        else:
            mse_global = np.mean((true - pred) ** 2)

        # 2. SỬA LỖI KHỐI ĐẶC XỊT (OVERLAPPING BUG)
        # Chỉ lấy điểm thời gian đầu tiên (index 0) của mỗi cửa sổ trượt để vẽ thành 1 đường liên tục
        if pred.ndim == 3:
            p_series = pred[:, 0, -1] 
            t_series = true[:, 0, -1]
        elif pred.ndim == 2:
            p_series = pred[:, -1]
            t_series = true[:, -1]
        else:
            p_series = pred
            t_series = true
            
        current_length = len(p_series)
        
        if expected_length is None:
            expected_length = current_length
            
        if current_length == expected_length:
            data_dict[model] = {
                'true': t_series,
                'pred': p_series,
                'mse': mse_global
            }
    else:
        print(f"[CẢNH BÁO] Không tìm thấy dữ liệu cho {model}.")

if not data_dict:
    print("Không có dữ liệu hợp lệ để vẽ biểu đồ.")
    exit()

# Tạo trục thời gian giả lập
dates = pd.date_range(start='2018-02-01', periods=expected_length, freq='H')

# Khởi tạo grid
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

for i, model in enumerate(models):
    ax = axes[i]
    label_letter = chr(97 + i)
    
    if model in data_dict:
        data = data_dict[model]
        
        # Vẽ 2 đường tín hiệu mượt mà
        ax.plot(dates, data['true'], label='Ground Truth', color='#7f7f7f', linewidth=1.2)
        ax.plot(dates, data['pred'], label='Prediction', color='#FFA500', linewidth=1.2, alpha=0.9)
        
        title_text = f"({label_letter}) {model} (MSE: {data['mse']:.4f})"
        ax.set_title(title_text, fontsize=12, pad=10, fontweight='bold')
        
        # SỬA LỖI TRỤC X ĐEN KỊT
        # Ép hiển thị theo Năm, với tối đa 4-5 nhãn để không bị rối
        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        formatter = mdates.DateFormatter('%m/%Y') # Chỉ hiển thị Năm
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        
        ax.grid(True, linestyle='-', alpha=0.3)
        ax.legend(loc='upper right', frameon=True, fontsize=9)
        
    else:
        ax.text(0.5, 0.5, f'Không có dữ liệu ({label_letter}) {model}', 
                ha='center', va='center', fontsize=12)

# Điều chỉnh layout thông minh
plt.tight_layout(pad=2.0)

# Lưu và hiển thị
plt.savefig('CA-MLP/experiment_figures/clean_sequence_comparison.png', dpi=300, bbox_inches='tight')
print("\nĐã lưu biểu đồ thành công: clean_sequence_comparison.png")
plt.show()