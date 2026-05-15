# 🚦 Xây dựng mô hình mạng nơ-ron tích chập (CNN) cho bài toán phân loại biển báo giao thông

Đồ án cuối kỳ — phân loại biển báo giao thông bằng mạng CNN **tự xây dựng** (không dùng pretrained) trên TensorFlow/Keras, thực nghiệm trên bộ dữ liệu chuẩn **GTSRB (German Traffic Sign Recognition Benchmark, 43 lớp)**, kèm demo web bằng Streamlit.

## Mục tiêu

- Xây dựng và huấn luyện một mô hình CNN từ đầu (không dùng pretrained) để phân loại 43 lớp biển báo GTSRB.
- Đánh giá mô hình bằng accuracy, classification report, confusion matrix.
- Triển khai demo web cho phép người dùng upload ảnh và nhận kết quả dự đoán.

## Cấu trúc project

```
final_project/
├── GTSRB.zip             # Dataset gốc (Kaggle format) — không commit
├── data/
│   ├── gtsrb_raw/        # Sinh ra khi giải nén GTSRB.zip (Train/, Test/, *.csv)
│   ├── classes.txt       # 43 mã lớp (GTSRB_00..GTSRB_42)
│   ├── classes_en.txt    # Tên 43 lớp tiếng Anh
│   ├── classes_vie.txt   # Tên 43 lớp tiếng Việt
│   └── processed/        # Ảnh đã crop theo class (sinh ra bởi prepare_data)
├── notebooks/
│   ├── colab_train.ipynb         # Train trên Colab T4 (GTSRB end-to-end)
│   └── colab_finetune_vn.ipynb   # Future work: fine-tune sang biển VN
├── src/
│   ├── config.py         # Hyperparameters & paths
│   ├── prepare_data.py   # GTSRB Kaggle → dataset classification (crop theo bbox CSV)
│   ├── data_loader.py    # Load dataset từ data/processed/
│   ├── preprocessing.py  # Augmentation
│   ├── model.py          # Kiến trúc CNN tự xây
│   ├── train.py          # Script huấn luyện
│   ├── evaluate.py       # Đánh giá chi tiết
│   └── predict.py        # Inference 1 ảnh
├── models/               # Lưu weights .keras + labels.json
├── reports/
│   ├── figures/          # Đồ thị, confusion matrix
│   └── snapshots/        # Sinh ra khi nhấn 📸 trong tab Camera realtime
├── app/
│   ├── streamlit_app.py  # Demo web (entry point) — tab Upload có UI crop ROI
│   └── realtime/         # Mode camera realtime (streamlit-webrtc)
│       ├── inference.py      # Cache model + hàm predict
│       ├── overlay.py        # Vẽ ROI box + nhãn lên frame
│       ├── video_processor.py# SignVideoProcessor (worker thread WebRTC)
│       └── tab.py            # UI sliders + nút snapshot
├── demo_images/          # Ảnh chuẩn bị sẵn cho buổi defense (3 tier)
│   ├── tier1_gtsrb/      # 10 ảnh GTSRB happy-path đã verify top-1 đúng
│   ├── tier2_vn_real/    # Ảnh biển VN thực tế (tự bổ sung)
│   └── tier3_failure/    # Failure case có chủ đích (biển VN không thuộc GTSRB)
├── requirements.txt
├── README.md
├── DEMO_SCRIPT.md           # Kịch bản 15 phút + Q&A cho buổi defense
├── PREPROCESSING.md         # Chi tiết pipeline tiền xử lý
├── MODEL_DEVELOPMENT.md     # Lịch sử phát triển model (VN → GTSRB)
├── REALTIME_CAMERA_SPEC.md  # Design spec camera realtime
├── REALTIME_CAMERA_DEV.md   # Dev log camera realtime
└── EXPORT_TFLITE.md         # Hướng dẫn export TFLite
```

## Cài đặt

```bash
# Tạo virtual environment
python -m venv .venv
.\.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate       # Linux/Mac

# Cài dependencies
pip install -r requirements.txt
```

## Quy trình chạy

### Bước 1 — Chuẩn bị dữ liệu

Dataset gốc là GTSRB Kaggle (Train/, Test/, `Train.csv`, `Test.csv`). Script sẽ crop theo bbox trong CSV, chia train/val/test và sinh dataset classification:

```bash
python -m src.prepare_data                          # dùng GTSRB.zip ở root
python -m src.prepare_data --raw data/gtsrb_raw     # dùng thư mục đã giải nén
```

Sinh ra `data/processed/{train,val,test}/<class_folder>/*.png`, `models/labels.json` và `reports/prepare_data_summary.json`.

### Bước 2 — Huấn luyện

**Cách 1 — Local**:

```bash
python -m src.train
```

**Cách 2 — Google Colab (khuyến nghị nếu không có GPU)**:

1. Zip toàn bộ folder dự án (kèm `data/processed/`) thành `final_project.zip`.
2. Mở [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) trên Colab.
3. Bật GPU (`Runtime → Change runtime type → GPU T4`).
4. Chạy lần lượt các cell — notebook tự upload zip, train, evaluate và download kết quả.

Kết quả:
- Model lưu tại `models/custom_cnn_v1.keras`
- Đồ thị learning curves: `reports/figures/training_curves.png`
- Log: `reports/training_log.csv`

### Bước 3 — Đánh giá

```bash
python -m src.evaluate
```

Sinh ra:
- `reports/classification_report.txt` (precision/recall/F1 từng lớp)
- `reports/figures/confusion_matrix.png`

### Bước 4 — Inference 1 ảnh

```bash
python -m src.predict --image path/to/sign.jpg
```

### Bước 5 — Demo web Streamlit

```bash
streamlit run app/streamlit_app.py
```

Trình duyệt sẽ tự mở tại `http://localhost:8501`. Sidebar có 2 chế độ:

- **📁 Upload ảnh** — chọn ảnh JPG/PNG. Mặc định bật toggle **"✂️ Crop ROI thủ công"**:
  kéo khung xanh trên ảnh để chọn vùng biển báo, preview + top-k kết quả cập nhật
  real-time ở cột bên phải. Có lựa chọn aspect ratio **Vuông 1:1** (mặc định, hợp
  với biển tròn/tam giác/vuông) hoặc **Tự do**. Tắt toggle để predict full-size
  (dùng để minh hoạ tầm quan trọng của ROI cropping). UI dùng
  [`streamlit-cropper`](https://github.com/turner-anderson/streamlit-cropper)
  — đã khai báo trong `requirements.txt`.
- **🎥 Camera realtime** — bật webcam, đưa biển báo vào khung ROI ở giữa
  hình. Tham số chỉnh được: kích thước ROI, ngưỡng confidence, số frame
  smoothing, hiển thị FPS. Nút **📸 Chụp snapshot** lưu khung hình hiện tại
  vào `reports/snapshots/YYYYMMDD_HHMMSS_<label>_<conf>.png`.

> Realtime dùng [`streamlit-webrtc`](https://github.com/whitphx/streamlit-webrtc).
> Lần đầu chạy, trình duyệt sẽ hỏi quyền camera. Model là **classifier**
> (không phải detector) nên mỗi frame chỉ phân loại 1 biển trong khung ROI.

> 💡 Trước buổi bảo vệ, đọc [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) để xem kịch bản
> 15 phút + bộ câu Q&A mẫu, và chuẩn bị ảnh demo theo
> [`demo_images/README.md`](demo_images/README.md).

## Kiến trúc CNN

3 khối Conv (32 → 64 → 128 filters), mỗi khối gồm:
`Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU → MaxPool → Dropout`,
sau đó `GlobalAvgPool → Dense(256) → BN → Dropout → Dense(num_classes, softmax)`.

Xem chi tiết tại [`src/model.py`](src/model.py).

## Hyperparameters chính

| Tham số | Giá trị |
|---|---|
| Image size | 48×48 RGB |
| Batch size | 128 |
| Epochs | 60 (EarlyStopping patience=12) |
| Optimizer | Adam, lr=1e-3 |
| Loss | Categorical Cross-Entropy |
| Augmentation | Rotation ±12°, zoom ±10%, translate ±6%, brightness, contrast, saturation |

Có thể chỉnh trong [`src/config.py`](src/config.py) và [`src/preprocessing.py`](src/preprocessing.py).

## Roadmap

- [x] Skeleton project
- [x] Tải dataset GTSRB (Kaggle format, 43 lớp)
- [x] Pipeline tiền xử lý: crop bbox theo CSV → classification
- [x] Train baseline CNN (đạt **97.09%** test accuracy, top-3 99.34% — xem [`MODEL_DEVELOPMENT.md`](MODEL_DEVELOPMENT.md))
- [x] Tune hyperparameters (class_weight cap, augmentation, EarlyStopping)
- [x] Demo Streamlit: tab Upload + interactive ROI crop UI
- [x] Camera realtime mode (`streamlit-webrtc`)
- [x] Bộ ảnh demo `demo_images/` + kịch bản `DEMO_SCRIPT.md`
- [ ] So sánh với VGG16/ResNet50 (transfer learning) — tùy chọn
- [ ] Fine-tune transfer sang biển VN (notebook đã chuẩn bị: `notebooks/colab_finetune_vn.ipynb`)
- [ ] Báo cáo & slide bảo vệ
