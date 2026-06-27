import os
import sys
import warnings
from types import SimpleNamespace

import torch
import torch.nn as nn

# Tắt cảnh báo của thop để terminal hiển thị sạch sẽ
warnings.filterwarnings("ignore", module="thop")

# Xử lý an toàn: Tránh crash toàn bộ hệ thống nếu môi trường chưa cài thop
try:
    from thop import profile as thop_profile
except ImportError:
    print("LỖI: Chưa cài đặt thư viện 'thop' để đo MACs.")
    print("Vui lòng chạy lệnh: pip install thop")
    sys.exit(1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.CA_MLP import Model


FREQ_TIME_DIM = {
    "h": 4,
    "t": 5,
    "s": 6,
    "m": 1,
    "a": 1,
    "w": 2,
    "d": 3,
    "b": 3,
}


COMMON_CONFIG = {
    "model": "CAW_KAN",
    "task_name": "long_term_forecast",
    "features": "M",
    "embed": "timeF",
    "freq": "h",
    "seq_len": 512,
    "label_len": 0,
    "pred_len": 96,
    "enc_in": 7,
    "dec_in": 7,
    "c_out": 7,
    "d_model": 16,
    "d_ff": 32,
    "factor": 1,
    "dropout": 0.1,
    "wavelet_type": "mexican_hat",
    "grid_size": 3.0,
    "channel_independence": 1,
    # Đã gỡ bỏ hoàn toàn batch_size khỏi cấu hình lõi của mô hình
}


DATASET_CONFIGS = [
    {
        "data": "ETTh1",
        "model_id": "ETTh1_96_96",
        "e_layers": 3,
        "num_wavelets": 0.5,
        "kernel_size": 3,
    },
    {
        "data": "ETTh2",
        "model_id": "ETTh2_96_96",
        "e_layers": 3,
        "num_wavelets": 0.5,
        "kernel_size": 3,
    },
    {
        "data": "ETTm1",
        "model_id": "ETTm1_96_96",
        "e_layers": 2,
        "freq": "t",
        "num_wavelets": 8,
        "kernel_size": 7,
    },
    {
        "data": "ETTm2",
        "model_id": "ETTm2_96_96",
        "e_layers": 2,
        "freq": "t",
        "num_wavelets": 0.5,
        "kernel_size": 7,
    },
    {
        "data": "weather",
        "model_id": "weather_96_96",
        "e_layers": 3,
        "freq": "t",
        "num_wavelets": 0.5,
        "kernel_size": 3,
        "enc_in": 21,
        "dec_in": 21,
        "c_out": 21,
    }
]


class ProfileWrapper(nn.Module):
    """Wrap CAW_KAN to profile a clean forecast forward pass."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x_enc, x_mark_enc):
        # Keep the same signature as the training path used by CAW_KAN.
        return self.model(x_enc, x_mark_enc, None, None)


def _to_namespace(config_dict):
    merged = dict(COMMON_CONFIG)
    merged.update(config_dict)
    return SimpleNamespace(**merged)


def _count_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


def _time_feature_dim(freq: str):
    freq_key = str(freq).lower()
    if freq_key in FREQ_TIME_DIM:
        return FREQ_TIME_DIM[freq_key]
    # Fallback compatible with many freq aliases (e.g., "15min", "3h").
    if freq_key.endswith("min"):
        return FREQ_TIME_DIM["t"]
    if freq_key and freq_key[-1] in FREQ_TIME_DIM:
        return FREQ_TIME_DIM[freq_key[-1]]
    return FREQ_TIME_DIM["h"]


def _build_dummy_x_mark(config, b_size=1):
    # Truyền trực tiếp b_size thay vì đọc từ config
    if getattr(config, "embed", "timeF") == "timeF":
        time_dim = _time_feature_dim(config.freq)
        return torch.rand(b_size, config.seq_len, time_dim)
    return torch.zeros(b_size, config.seq_len, 5, dtype=torch.long)


def profile_one(config):
    model = Model(config)
    model.eval()

    wrapped_model = ProfileWrapper(model)
    
    # Ép buộc đo lường chuẩn xác trên 1 mẫu duy nhất
    b_size = 1 
    
    dummy_x = torch.randn(b_size, config.seq_len, config.enc_in)
    dummy_x_mark = _build_dummy_x_mark(config, b_size)

    # 1. Đo lường số lượng tham số (Parameters)
    total_params = _count_params(model)

    # 2. Đo lường MACs bằng thư viện thop
    with torch.no_grad():
        macs, _ = thop_profile(wrapped_model, inputs=(dummy_x, dummy_x_mark), verbose=False)

    return {
        "dataset": config.data,
        "params_total": total_params,
        "macs_total": macs,
    }


def print_results(rows):
    # Cột Batch đã được loại bỏ hoàn toàn
    header = (
        f"{'Dataset':<10} | {'Params (Total)':>16} | {'MACs (Total)':>18} | {'MACs (G)':>10}"
    )
    separator = "-" * len(header)
    
    print(separator)
    print(header)
    print(separator)
    
    for row in rows:
        g_macs = row['macs_total'] / 1e9  # Chuyển đổi sang Giga-MACs
        print(
            f"{row['dataset']:<10} | "
            f"{row['params_total']:>16,} | "
            f"{int(row['macs_total']):>18,} | "
            f"{g_macs:>10.4f}"
        )
    print(separator)


def main():
    rows = []
    for cfg in DATASET_CONFIGS:
        config = _to_namespace(cfg)
        rows.append(profile_one(config))
    print_results(rows)


if __name__ == "__main__":
    main()