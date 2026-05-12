# 🚦 Nhận diện biển báo giao thông Việt Nam bằng CNN tự xây

Đồ án cuối kỳ — phân loại biển báo giao thông Việt Nam sử dụng mạng nơ-ron tích chập (CNN) tự xây dựng bằng TensorFlow/Keras, kèm demo web bằng Streamlit.

## Mục tiêu

- Xây dựng và huấn luyện một mô hình CNN từ đầu (không dùng pretrained) để phân loại biển báo giao thông Việt Nam.
- Đánh giá mô hình bằng accuracy, classification report, confusion matrix.
- Triển khai demo web cho phép người dùng upload ảnh và nhận kết quả dự đoán.

## Cấu trúc project

```
final_project/
├── data/
│   ├── images/           # Ảnh gốc (YOLO format)
│   ├── labels/           # Nhãn YOLO .txt
│   ├── split_dataset/    # train_files.txt / test_files.txt
│   ├── classes*.txt      # Tên 52 lớp (code / EN / VI)
│   ├── processed/        # Ảnh đã crop theo class (sinh ra bởi prepare_data)
│   └── README.md
├── notebooks/            # Notebook EDA và thử nghiệm
├── src/
│   ├── config.py         # Hyperparameters & paths
│   ├── prepare_data.py   # Crop bbox YOLO → dataset classification
│   ├── data_loader.py    # Load dataset từ data/processed/
│   ├── preprocessing.py  # Augmentation
│   ├── model.py          # Kiến trúc CNN tự xây
│   ├── train.py          # Script huấn luyện
│   ├── evaluate.py       # Đánh giá chi tiết
│   └── predict.py        # Inference 1 ảnh
├── models/               # Lưu weights .keras + labels.json
├── reports/
│   └── figures/          # Đồ thị, confusion matrix
├── app/
│   └── streamlit_app.py  # Demo web
├── requirements.txt
└── README.md
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

Dataset gốc có format YOLO (detection). Cần crop bounding box → dataset classification:

```bash
python -m src.prepare_data
```

Sinh ra `data/processed/{train,val,test}/<class_folder>/*.jpg` và `models/labels.json`.
Xem chi tiết tại [`data/README.md`](data/README.md).

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

Trình duyệt sẽ tự mở tại `http://localhost:8501`.

## Kiến trúc CNN

3 khối Conv (32 → 64 → 128 filters), mỗi khối gồm:
`Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU → MaxPool → Dropout`,
sau đó `GlobalAvgPool → Dense(256) → BN → Dropout → Dense(num_classes, softmax)`.

Xem chi tiết tại [`src/model.py`](src/model.py).

## Hyperparameters chính

| Tham số | Giá trị |
|---|---|
| Image size | 48×48 RGB |
| Batch size | 64 |
| Epochs | 40 (EarlyStopping patience=8) |
| Optimizer | Adam, lr=1e-3 |
| Loss | Categorical Cross-Entropy |
| Augmentation | Rotation ±18°, zoom, translate, brightness, contrast |

Có thể chỉnh trong [`src/config.py`](src/config.py).

## Roadmap

- [x] Skeleton project
- [x] Tải dataset (YOLO format, 3,216 ảnh, 52 lớp)
- [x] Pipeline tiền xử lý: crop bbox → classification
- [ ] Train baseline CNN
- [ ] Tune hyperparameters
- [ ] So sánh với VGG16/ResNet50 (transfer learning) — tùy chọn
- [ ] Hoàn thiện demo Streamlit
- [ ] Báo cáo & slide bảo vệ
