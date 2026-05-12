# 🚦 Quá trình xây dựng & tinh chỉnh model

Tài liệu hợp nhất lịch sử phát triển model phân loại biển báo giao thông —
từ phiên bản đầu (dataset Việt Nam, bị **collapse**), qua bước migration
sang **GTSRB**, đến các vòng tinh chỉnh nâng cao accuracy.

> File này thay thế `CHANGES.md` cũ, gộp cả nội dung lịch sử Đợt 1.

---

## 📌 Tổng quan

| Mục | Lựa chọn cuối |
|---|---|
| Dataset | **GTSRB** (German Traffic Sign Recognition Benchmark, 43 lớp) |
| Input size | 48×48 RGB |
| Architecture | Custom CNN (`custom_cnn_v1`) |
| Optimizer | Adam, LR=1e-3 + ReduceLROnPlateau |
| Augmentation | Affine + brightness/contrast/saturation (xem `PREPROCESSING.md`) |
| Class weights | `compute_class_weight` cap ở **2.0** |
| Epochs | **60** + EarlyStopping patience=12 |

---

## Phase 0 — Dataset Việt Nam (đã loại bỏ)

### 0.1 Tình trạng ban đầu

Train trên dataset biển báo VN (50 lớp, scrape từ web) cho kết quả **collapse**:

| Chỉ số | Giá trị |
|---|---|
| Test accuracy | **0.53%** (thấp hơn random 1/50 = 2%) |
| Train accuracy | 1–2% (model gần như không học) |
| Loss | bám quanh 4.2 ≈ -ln(1/50), đứng yên |

### 0.2 Chẩn đoán

| # | Nguyên nhân | Bằng chứng |
|---|---|---|
| (a) | **Class imbalance cực đoan** | Lớp lớn nhất 629 ảnh, nhỏ nhất 4 ảnh → tỉ lệ 1:160 |
| (b) | **Ảnh 48×48 quá nhỏ** | Sau 3 MaxPool còn 6×6, không phân biệt được P_127_40/50/60/80 |
| (c) | **Augmentation không hoạt động** | `tf.keras.Sequential([Random*Layer])` qua `tf.data.map` mặc định `training=False` → không augment |
| (d) | **Lớp <30 sample không học được** | `MIN_SAMPLES_PER_CLASS=5` quá thấp |

### 0.3 Fix Đợt 1 (đã apply)

| File | Thay đổi |
|---|---|
| `src/config.py` | `IMG_SIZE` 48 → 64 |
| `src/prepare_data.py` | `MIN_SAMPLES_PER_CLASS` 5 → 30 (loại 17/52 lớp) |
| `src/preprocessing.py` | Viết lại `augment()` bằng `tf.image.*` + `ImageProjectiveTransformV3` (random thật) |
| `src/train.py` | Thêm `compute_class_weights(..., max_weight=5.0)` cap weight |

### 0.4 Kết quả Đợt 1

| Trục | Trước | Sau |
|---|---|---|
| Số lớp | 50 | **35** (loại 17 lớp <30 sample) |
| Train acc cuối | sụp đổ | **97.2%** |
| Val acc cuối | sụp đổ | **98.7%** |
| **Test accuracy** | **0.53%** | **98.44%** |
| Test top-3 | 4.8% | **99.86%** |

→ Vượt kỳ vọng nhưng **chỉ phủ 35/50 lớp** (17 lớp hiếm bị loại). Đây là động lực
để chuyển sang dataset chuẩn quốc tế.

---

## Phase 1 — Migration sang GTSRB

### 1.1 Lý do chuyển

| Vấn đề dataset VN | GTSRB giải quyết |
|---|---|
| Chỉ 35/50 lớp có đủ data, mất 17 biển hiếm | **43 lớp**, lớp ít nhất vẫn có **210** ảnh train |
| Imbalance 1:160 → buộc phải cap class_weight mạnh | Imbalance ~1:11 (210–2250 ảnh/lớp) |
| Crowdsourced, chất lượng ảnh không đồng đều | Captured từ video xe chạy thực tế, có ROI bbox sẵn |
| Không có test set chuẩn | **12,630** ảnh test riêng biệt, có ground truth |
| Khó so sánh với benchmark | GTSRB là benchmark chuẩn, dễ đối chiếu SOTA |

### 1.2 Thay đổi pipeline

| File | Trước (VN) | Sau (GTSRB) |
|---|---|---|
| `src/config.py` | `IMG_SIZE=64`, paths VN | `IMG_SIZE=48`, paths `gtsrb_raw/`, `processed/` |
| `src/prepare_data.py` | Quét folder → split 70/15/15 | Giải nén zip → đọc `Train.csv`/`Test.csv` → ROI crop + 5% padding → stratified split 90/10 cho train/val |
| `data/classes.txt` (+ `_en`, `_vie`) | 50 lớp VN | **43 lớp** GTSRB (ID, EN, VIE) |
| `notebooks/colab_train.ipynb` | — | Thêm section Kaggle API + Drive + Upload |
| `PREPROCESSING.md` | — | Mới: tài liệu hóa pipeline 3 lớp |

### 1.3 Tại sao quay về 48×48?

GTSRB vốn là benchmark **48×48** (nhiều paper SOTA cùng dùng). Ảnh đã được crop ROI
chặt nên 48×48 đủ thông tin: con số/ký hiệu rõ ràng, không pha lẫn background như
dataset VN. Giảm size cũng cho phép `BATCH_SIZE=128` thoải mái trên Colab T4.

---

## Phase 2 — Train lần đầu trên GTSRB

### 2.1 Cấu hình

| Hyperparameter | Giá trị |
|---|---|
| EPOCHS | 30 |
| BATCH_SIZE | 128 |
| LEARNING_RATE | 1e-3 (+ ReduceLROnPlateau ×0.5) |
| EARLY_STOPPING_PATIENCE | 8 |
| class_weight cap | 5.0 |

### 2.2 Kết quả

| Metric | Giá trị | Kỳ vọng GTSRB | Đánh giá |
|---|---|---|---|
| **Test accuracy** | **89.78%** | 95–97% | ⚠️ Thấp ~6–7% |
| Test top-3 | 97.24% | ~99% | ✅ OK |
| Val accuracy (epoch 29) | 93.70% | — | ⚠️ Còn đang tăng |
| Train accuracy (epoch 29) | 93.53% | — | ✅ Không overfit |


### 2.3 Chẩn đoán 3 vấn đề

#### ❌ Vấn đề A — Model chưa hội tụ (under-trained)

Đọc `reports/training_log.csv`:

```
epoch  val_acc   val_loss
26     0.86      0.42
27     0.92      0.26
28     0.90      0.29
29     0.94      0.20   ← vẫn đang tăng/giảm rất nhanh
```

Val_loss vẫn dốc xuống ở epoch cuối → **EarlyStopping không kích hoạt**, model bị
cắt training giữa chừng vì hết epochs.

#### ❌ Vấn đề B — Vài epoch đầu loạn

```
epoch 0:  train_acc=0.21, val_loss=5.67
epoch 1:  train_acc=0.16, val_loss=4.53
epoch 2:  train_acc=0.13                   ← acc GIẢM
epoch 5:  train_acc=0.13  (LR ↓ 5e-4)
epoch 7:  train_acc=0.24                   ← mới bắt đầu học
```

Mất ~7 epoch mới ổn định. Nghi ngờ do **class_weight=5.0 quá mạnh** kết hợp
LR=1e-3 → gradient lớn ở các batch chứa lớp hiếm (weight 5×).

#### ❌ Vấn đề C — Mất cân bằng precision/recall

Model **over-predict** lớp ít sample:

| Lớp | Precision | Recall | Diễn giải |
|---|---|---|---|
| Xe đạp qua đường | **0.28** | 1.00 | Gắn nhãn này cho rất nhiều ảnh sai |
| Rẽ trái phía trước | **0.40** | 1.00 | Tương tự |
| Cẩn thận băng/tuyết | **0.58** | 0.98 | — |
| Đi thẳng hoặc rẽ trái | **0.60** | 1.00 | — |

Và **under-predict** lớp speed limit:

| Lớp | Precision | Recall |
|---|---|---|
| Speed limit (20km/h) | 1.00 | **0.73** |
| Speed limit (30km/h) | 1.00 | **0.77** |
| Speed limit (50km/h) | 1.00 | **0.66** |

→ `class_weight` cap=5.0 đẩy quá mạnh về phía minority classes; majority classes
(speed limits) bị "bỏ rơi".

---

## Phase 3 — Tinh chỉnh (Fix #1 + #2, đang áp dụng)

### 3.1 Fix #1 — Train lâu hơn để hội tụ

**File:** `src/config.py`

| Tham số | Cũ | Mới | Lý do |
|---|---|---|---|
| `EPOCHS` | 30 | **60** | Phase 2 cho thấy val acc còn tăng ở epoch 29 |
| `EARLY_STOPPING_PATIENCE` | 8 | **12** | Tránh dừng sớm khi LR vừa giảm xong |

Kỳ vọng: model tự dừng quanh epoch 40–50 khi val_loss thật sự plateau, đẩy test
accuracy lên 94–96%.

### 3.2 Fix #2 — Giảm class_weight cap

**File:** `src/train.py`, hàm `compute_class_weights`

```python
def compute_class_weights(y_train, num_classes, max_weight=2.0):
    ...
```

| Tham số | Cũ | Mới |
|---|---|---|
| `max_weight` | 5.0 | **2.0** |

Lý do:
- GTSRB cân bằng vừa phải (1:11), không cần cap mạnh như dataset VN cũ (1:160)
- Cap 5.0 đã chứng minh gây over-prediction (precision 0.28 ở Xe đạp)
- Cap 2.0 vẫn cho minority classes weight ~2× majority — đủ để không bị bỏ qua,
  không quá mạnh để bias dự đoán

Kỳ vọng: precision của 4 lớp problematic (Bicycles, Turn left, Ice/snow, Go
straight or left) tăng từ 0.28–0.60 lên 0.80+; recall của speed limits tăng từ
0.66–0.77 lên 0.85+.

### 3.3 Fix CHƯA áp dụng (dự phòng cho Phase 4)

| # | Fix | Khi nào dùng |
|---|---|---|
| 3 | `LEARNING_RATE` 1e-3 → 5e-4 + warmup 3 epoch | Nếu epoch 0–7 vẫn loạn sau Fix #1+#2 |
| 4 | `BATCH_SIZE` 128 → 256 | Tăng tốc training (T4 đủ VRAM cho 48×48) |
| 5 | Bỏ `class_weight` hoàn toàn | Nếu Fix #2 vẫn over-correct minorities |
| 6 | LR cosine decay thay ReduceLROnPlateau | Stable hơn ở epoch cuối |

---

## Phase 4 — Kết quả train lại với Fix #1+#2

### 4.1 Tổng kết metric (test set 12,630 ảnh GTSRB)

| Metric | Phase 2 (v1) | Phase 4 (v2) | Δ |
|---|---|---|---|
| **Test accuracy** | 89.78% | **97.09%** | **+7.31 pp** ✅ |
| Test top-3 | 97.24% | **99.34%** | +2.10 pp |
| Test loss | — | **0.1108** | — |
| Macro-F1 | — | **0.9617** | — |
| Weighted-F1 | — | **0.9711** | — |

→ **Vượt mục tiêu ≥95%**. Không cần áp Fix #3–#6 dự phòng.

### 4.2 Hành vi training

Đọc `reports/training_log.csv` (60 epochs, không EarlyStopping):

| Mốc | Epoch | val_acc | val_loss | LR | Ghi chú |
|---|---|---|---|---|---|
| Khởi động loạn | 0–7 | 0.01–0.21 | 7.5 → 4.1 | 1e-3 → 5e-4 | Vẫn 7 epoch warm-up "chậm" như Phase 2 |
| Bứt phá | 14–20 | 0.34 → 0.62 | 2.59 → 1.23 | 2.5e-4 | ReduceLR lần 2 (ep 10) ổn định lại |
| Ổn định cao | 30–46 | 0.81 → 0.96 | 0.32 → 0.12 | 2.5e-4 | Train/val sát nhau, không overfit |
| Hội tụ | 47–59 | 0.97 → **0.991** | 0.084 → **0.032** | 1.25e-4 | LR giảm lần 3 (ep 47) tinh chỉnh nốt |
| **Cuối** (ep 59) | — | **0.9911** | **0.0319** | 1.25e-4 | train_acc 0.9742 → không overfit |

Best val_acc = **0.9923** (ep 57). EarlyStopping không trigger vì val_loss vẫn dao động giảm.

### 4.3 Verify 4 lớp "problematic" của Phase 2

| Lớp | Precision (P2 → P4) | Recall (P2 → P4) | Đánh giá |
|---|---|---|---|
| Xe đạp qua đường | 0.28 → **0.93** | 1.00 → 1.00 | ✅ Đẩy precision +0.65 |
| Rẽ trái phía trước | 0.40 → **0.95** | 1.00 → 1.00 | ✅ Đẩy precision +0.55 |
| Cẩn thận băng/tuyết | 0.58 → **0.85** | 0.98 → 0.98 | ✅ +0.27 |
| Đi thẳng hoặc rẽ trái | 0.60 → **0.98** | 1.00 → 1.00 | ✅ +0.38 |

Và recall các speed limit hồi phục:

| Lớp | Recall (P2 → P4) | Precision (P2 → P4) |
|---|---|---|
| Speed limit (20km/h) | 0.73 → **0.95** | 1.00 → 1.00 |
| Speed limit (30km/h) | 0.77 → **0.97** | 1.00 → 0.99 |
| Speed limit (50km/h) | 0.66 → **0.98** | 1.00 → 1.00 |

→ Cả 2 hướng (over-predict minorities + under-predict speed limits) đã được giải quyết.
Việc giảm `class_weight` cap 5.0 → 2.0 đúng hướng.

### 4.4 Vấn đề còn tồn (cho Phase 5)

Một số lớp f1 < 0.92 — chủ yếu là **under-recall** (model "kén chọn" hơn ở các lớp
hiếm/mịn) hoặc **over-precision sụt** ở 2 lớp đặc thù:

| Lớp | Precision | Recall | F1 | Hướng vấn đề |
|---|---|---|---|---|
| Bắt buộc đi vòng xuyến | **0.67** | 0.97 | 0.79 | Over-predict (false positive cao) |
| Hết đoạn cấm vượt với xe >3.5t | **0.76** | 0.86 | 0.81 | Cả P và R đều thấp — dễ nhầm với lớp 41 |
| Giới hạn tốc độ (60km/h) | 1.00 | **0.83** | 0.91 | Under-predict — nhầm sang 50/80 |
| Hết giới hạn tốc độ (80km/h) | 1.00 | **0.83** | 0.91 | Under-predict |
| Đường gồ ghề | 0.99 | **0.83** | 0.90 | Under-predict |
| Giới hạn tốc độ (120km/h) | 0.99 | **0.89** | 0.94 | Under-predict |

Số lượng lớp này nhỏ (6/43) và đều >0.79 f1 — không ảnh hưởng bottom-line nhưng
là vùng cải thiện rõ ràng nhất.

### 4.5 Roadmap Phase 5 (tùy chọn, nếu muốn ≥98%)

1. **Tách 2 cặp dễ nhầm** (40 vs 41/42, 03/05 vs 06/08) bằng confusion matrix
   → áp dụng **mixup/cutmix** tập trung cho cặp này
2. **Test-time augmentation** (TTA): predict 5 crop + flip → vote → kỳ vọng +0.3–0.5 pp
3. **Transfer learning** MobileNetV2 / EfficientNetB0 (input 96×96) — chỉ thử nếu cần
   ≥98% cho yêu cầu khắt khe hơn
4. **Hiện tại**: model đủ tốt cho app Streamlit demo → ưu tiên ship

---

## 🔁 Cách chạy lại với cấu hình mới

### Trên Colab (khuyên dùng)

```python
# Mở notebooks/colab_train.ipynb, chạy tuần tự:
# - Section 2: upload code zip
# - Section 4A: tải GTSRB qua Kaggle API
# - Section 5: prepare_data
# - Section 6: train (TỰ ĐỘNG dùng EPOCHS=60, max_weight=2.0 mới)
# - Section 7: evaluate + download model
```

### Local (nếu có GPU)

```bash
py -m src.prepare_data        # tạo data/processed/ (skip nếu đã có)
py -m src.train               # train ~45–60 phút trên T4
py -m src.evaluate            # ghi metrics.json + classification_report
```

---

## ↩️ Rollback nếu Fix #1+#2 cho kết quả tệ hơn

```bash
git diff HEAD src/config.py src/train.py
git checkout HEAD -- src/config.py src/train.py
```

Hoặc thủ công:

| File | Hoàn về |
|---|---|
| `src/config.py` | `EPOCHS = 30`, `EARLY_STOPPING_PATIENCE = 8` |
| `src/train.py` | `max_weight: float = 5.0` |

---

## 📚 File liên quan

| File | Nội dung |
|---|---|
| `PREPROCESSING.md` | Pipeline tiền xử lý 3 lớp (offline, load-time, augment) |
| `src/config.py` | Tất cả hyperparameter tập trung |
| `src/train.py` | Training loop + class_weight + callbacks |
| `src/prepare_data.py` | Crop ROI từ CSV + stratified split |
| `notebooks/colab_train.ipynb` | Workflow Colab end-to-end |
| `reports/metrics.json` | Test accuracy/loss/top3 lần train gần nhất |
| `reports/classification_report.txt` | Per-class precision/recall/f1 |
| `reports/training_log.csv` | Log epoch-by-epoch train/val |

---

## 📅 Lịch sử versions

| Phase | Dataset | Test acc | Notes |
|---|---|---|---|
| 0 (initial) | VN 50 lớp | **0.53%** | Collapse |
| 0 (Đợt 1 fix) | VN 35 lớp | **98.44%** | Mất 17 lớp hiếm |
| 1–2 (GTSRB v1) | GTSRB 43 lớp | **89.78%** | Under-trained + over-weighted |
| 3–4 (GTSRB v2) | GTSRB 43 lớp | **97.09%** | Fix #1+#2 — đạt mục tiêu ≥95% (top-3 99.34%) |
| **5 (tùy chọn)** | GTSRB 43 lớp | (chưa train) | TTA / mixup / transfer learning nếu cần ≥98% |
