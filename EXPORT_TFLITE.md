# 📦 Export Keras → TFLite cho app Android

Tài liệu ghi lại quá trình chuyển đổi `models/custom_cnn_v1.keras` (CNN tự xây, 43 lớp GTSRB) sang TensorFlow Lite để bundle vào app Android (repo riêng tại `d:\IT-HAU\android\`).

## 1. Mục tiêu

- Sinh `.tflite` chạy on-device, không cần Internet, dưới 2 MB.
- Cho ra **nhiều biến thể** (FP32 / FP16 / INT8) để có thể chọn theo trade-off **độ chính xác ↔ kích thước ↔ tốc độ**.
- Tự verify rằng `.tflite` cho prediction giống Keras gốc trước khi ship.
- Sinh kèm `labels_android.json` rút gọn cho app dùng.

## 2. Script

[`scripts/export_tflite.py`](scripts/export_tflite.py) — chạy đứng độc lập, không sửa code training.

```bash
# Mặc định: thử cả 3 biến thể, INT8 sẽ tự skip nếu thiếu data
python -m scripts.export_tflite

# Chỉ FP16 (khuyến nghị làm default ship cho app)
python -m scripts.export_tflite --variants fp16

# INT8 với 300 ảnh calibration
python -m scripts.export_tflite --variants int8 --rep-samples 300
```

CLI flags:

| Flag | Default | Ý nghĩa |
|---|---|---|
| `--variants` | `fp32 fp16 int8` | Subset biến thể cần xuất |
| `--rep-samples` | 200 | Số ảnh từ `data/processed/val/` cho INT8 representative dataset |
| `--verify-samples` | 200 | Số ảnh từ `data/processed/test/` để so prediction TFLite vs Keras |

## 3. Output

Tất cả nằm trong `models/tflite/`:

```
models/tflite/
├── custom_cnn_v1_fp32.tflite     # Bản full precision
├── custom_cnn_v1_fp16.tflite     # Bản FP16 — recommended default
├── custom_cnn_v1_int8.tflite     # Bản INT8 (chỉ khi đủ data)
├── labels_android.json           # 43 lớp, chỉ giữ idx + name_vi + name_en
└── export_report.json            # Kết quả verify
```

## 4. Kết quả lần chạy đầu (2026-05-13)

Môi trường: Windows, Python 3.13.7, TensorFlow 2.21.0, CPU only.

```
>>> Loading Keras model: models/custom_cnn_v1.keras
    Output shape: (None, 43), params: 333,899
    Representative samples (val): 0
    Verify samples (test):        0

>>> Converting...
[ok]   FP32  →  custom_cnn_v1_fp32.tflite  (1302.1 KB)
[ok]   FP16  →  custom_cnn_v1_fp16.tflite  (659.6 KB)
[skip] INT8 cần representative dataset (data/processed/val/) — bỏ qua.

>>> Verifying...
    {'path': 'custom_cnn_v1_fp32.tflite', 'skipped': True}
    {'path': 'custom_cnn_v1_fp16.tflite', 'skipped': True}
[ok]   labels →  labels_android.json  (43 classes)
```

| Biến thể | Size | Giảm so với FP32 | Ghi chú |
|---|---:|---:|---|
| FP32 | 1302.1 KB | — | Baseline, dùng để verify |
| FP16 | 659.6 KB | **−49.3%** | Default ship cho app, hỗ trợ GPU delegate tốt |
| INT8 | _N/A_ | _N/A_ | Cần chạy lại sau khi có `data/processed/` |

> **Lý do INT8 + verify bị skip**: thư mục `data/processed/` đang rỗng vì chưa giải nén `GTSRB.zip` và chạy `python -m src.prepare_data` trên máy này. Sau khi prepare data, chạy lại script là có đủ.

## 5. Tiền xử lý (BẮT BUỘC khớp khi inference)

Tham chiếu `src/preprocessing.py::preprocess_single_image`. App Android phải làm **đúng**:

1. Convert ảnh sang RGB.
2. Resize **bilinear** về **48×48**.
3. Cast sang float32, **chia 255.0** (range [0, 1]).
4. **KHÔNG** mean/std normalization, **KHÔNG** flip ngang.
5. Tensor input shape: `[1, 48, 48, 3]`.

Với INT8 model, thay bước 3 bằng quantize: `q = clip(x/scale + zero_point, 0, 255)` lấy `scale, zero_point` từ `interpreter.getInputTensor(0).quantizationParams()`.

## 6. Cách lấy artifact sang repo Android

Repo Android nằm tại `d:\IT-HAU\android\`. Sau khi xuất xong:

```powershell
$src = "d:\IT-HAU\KHMT\final_project\models\tflite"
$dst = "d:\IT-HAU\android\app\src\main\assets"   # hoặc đường dẫn assets tương ứng
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Copy-Item "$src\custom_cnn_v1_fp16.tflite" $dst
Copy-Item "$src\labels_android.json"       $dst
```

## 7. Kế hoạch tiếp theo (khi đã có `data/processed/`)

1. `python -m src.prepare_data` (giải nén `GTSRB.zip` nếu cần).
2. Chạy lại `python -m scripts.export_tflite` → có thêm `int8.tflite` + `export_report.json` đầy đủ.
3. Xem `match_with_keras_pct` trong `export_report.json`:
   - **≥ 99%**: ship FP16/INT8 thoải mái.
   - **95–99%**: chỉ ship FP16, giữ INT8 như tuỳ chọn.
   - **< 95%**: dừng lại, kiểm tra preprocessing/calibration set.

## 8. Troubleshooting

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `Tee-Object` báo NativeCommandError | TF in WARNING ra stderr → PowerShell hiểu nhầm là exception | Bỏ qua, đọc stdout — script vẫn chạy đúng |
| `Could not find Keras model` | Chưa train | Chạy `python -m src.train` trước |
| INT8 bị skip dù đã có data | `data/processed/val/` rỗng (chưa split val) | Chạy lại `prepare_data.py`, kiểm tra `VAL_RATIO_FROM_TRAIN` |
| Verify match < 95% | Calibration set không đại diện hoặc preprocessing lệch | Tăng `--rep-samples`, đảm bảo ảnh val đa dạng đủ 43 lớp |
