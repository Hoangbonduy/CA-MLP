## Cài đặt môi trường

```bash
cd CA_MLP
conda create -n cawkan python=3.12.13 -y
conda activate camlp
pip install -r requirements.txt
```

## Dataset

Dataset lấy từ [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy).

Sau khi tải xong, đặt dữ liệu vào thư mục `dataset` của project.

## Chạy các dữ liệu

Ví dụ:
### Etth1 với độ dài dự báo 96

```bash
bash scripts/Etth1/etth1_96.sh
```

### Etth1 với độ dài dự báo 96 không dùng gpu
```bash
bash scripts/Etth1/etth1_96.sh --no_use_gpu
```

## Chạy tham số và số lượng MACs của mô hình:
```bash
python CA-MLP/utils/parameter.py
```

## Các thư mục trong scripts:
- Etth1, Etth2, Ettm1, Ettm2, Weather, Exchange: Các file chạy các dữ liệu của mô hình
- hiddendim (kích thước lớp ẩn trong lớp MLP)
- kernelsize (kích thước kênh trong lớp nhận thức ngữ cảnh (lớp tích chập 1 chiều))
- numblocks (số lượng khối CA-MLP)
- Phân tích loại bỏ
   + nomlp: Loại bỏ lớp MLP
   + noconv1d: Loại bỏ lớp tích chập 1 chiều
   + noresidualconnection: Loại bỏ kết nối phần dư