# 📊 Tiền xử lý dữ liệu GTSRB

Tài liệu mô tả toàn bộ pipeline tiền xử lý từ raw `GTSRB.zip` đến input của
model CNN. Pipeline gồm **3 lớp** chạy ở **2 thời điểm**:

| Lớp | Chạy khi nào | Module | Output |
|---|---|---|---|
| 1. Tiền xử lý ngoại tuyến | 1 lần (chạy script) | `src/prepare_data.py` | `data/processed/{train,val,test}/` |
| 2. Load + decode + resize | Mỗi batch (real-time) | `src/data_loader.py` | Tensor `[B, 48, 48, 3]` ∈ `[0,1]` |
| 3. Data augmentation | Mỗi batch train (real-time) | `src/preprocessing.py` | Tensor đã augment |

```
GTSRB.zip
   │ (1) prepare_data.py
   ▼
data/processed/{train,val,test}/<class_folder>/<image>.png
   │ (2) data_loader.py
   ▼
tf.data.Dataset (resize + normalize)
   │ (3) preprocessing.augment  ← chỉ áp dụng cho train
   ▼
Model CNN
```

---

## 🔧 Lớp 1 — Tiền xử lý ngoại tuyến (`src/prepare_data.py`)

Script chạy 1 lần để biến raw GTSRB → dataset classification chuẩn.

### Cách chạy

```bash
python -m src.prepare_data                          # mặc định: GTSRB.zip ở root
python -m src.prepare_data --raw data/gtsrb_raw     # đã giải nén sẵn
python -m src.prepare_data --zip path/to/zip
```

### Các bước & hàm tương ứng

#### Bước 1.1 — Đọc metadata 43 lớp

**Hàm:** `load_class_metadata()` (dòng 52)

Đọc 3 file `data/classes.txt`, `classes_en.txt`, `classes_vie.txt` (mỗi file
43 dòng) và build list dict:

```python
{
  "idx": 14, "code": "GTSRB_14",
  "name_en": "Stop", "name_vi": "Dừng lại",
  "folder": "14_Stop"           # tên folder ASCII-safe sau slugify
}
```

Thứ tự = thứ tự dòng trong file = `ClassId` của GTSRB (0-42).

**Helper:** `slugify(text)` (dòng 40) — chuẩn hóa Unicode → ASCII, thay ký tự
đặc biệt bằng `_`. Ví dụ `"Speed limit (20km/h)"` → `"Speed_limit_20km_h"`.

#### Bước 1.2 — Giải nén ZIP (nếu cần)

**Hàm:** `ensure_extracted(zip_path, raw_dir)` (dòng 69)

- Kiểm tra `data/gtsrb_raw/` đã có đủ `Train/`, `Test/`, `Train.csv`, `Test.csv` chưa
- Nếu chưa → giải nén `GTSRB.zip` (~611MB → ~720MB ảnh PNG)
- Nếu có rồi → bỏ qua (không giải nén lại)

#### Bước 1.3 — Đọc CSV

**Hàm:** `parse_csv(csv_path)` (dòng 87)

Đọc `Train.csv` / `Test.csv` lấy 7 cột:

| Cột CSV | Field dict | Ý nghĩa |
|---|---|---|
| `ClassId` | `class_id` | 0-42 |
| `Path` | `path` | vd `Train/14/00014_00000_00000.png` |
| `Roi.X1`, `Roi.Y1` | `x1`, `y1` | top-left của bbox biển báo |
| `Roi.X2`, `Roi.Y2` | `x2`, `y2` | bottom-right |

> 💡 **Tại sao có ROI?** Ảnh GTSRB gốc có **viền ~10%** xung quanh biển
> (background, cây cối). ROI cho biết vùng pixel chính xác chứa biển báo.

#### Bước 1.4 — Stratified train/val split

**Hàm:** `stratified_train_val_split(rows, val_ratio, seed)` (dòng 102)

- Group `train_rows` theo `class_id` → 43 nhóm
- Mỗi nhóm: shuffle (seed cố định) → lấy 10% đầu làm val, 90% còn lại làm train
- **Đảm bảo mọi lớp đều xuất hiện ở val** (kể cả lớp ít sample nhất)
- `Test.csv` không bị split — dùng nguyên làm test set

Kết quả ước lượng: train ~35,288 | val ~3,921 | test 12,630.

#### Bước 1.5 — Crop ROI + lưu ảnh

**Hàm:** `crop_with_roi(img, x1, y1, x2, y2)` (dòng 118)
       + `process_split(rows, split_name, raw_dir, classes)` (dòng 127)

Mỗi ảnh:
1. Mở bằng PIL → convert RGB
2. Crop bbox `(x1, y1) → (x2, y2)` **kèm padding 5%** mỗi cạnh
   (`PADDING_RATIO = 0.05`)
3. Lưu PNG vào `data/processed/<split>/<folder_lớp>/<basename>.png`

> 💡 **Padding 5%** giữ lại 1 chút context xung quanh biển → robust hơn
> với bbox detection chưa sát ở inference time.

#### Bước 1.6 — Dọn dữ liệu cũ

**Hàm:** `clean_processed_dir()` (dòng 151)

Xóa toàn bộ `data/processed/{train,val,test}` cũ trước khi ghi mới — tránh
trộn lẫn dataset cũ và mới.

#### Bước 1.7 — Sinh metadata + summary

- `models/labels.json` — list 43 lớp (cho inference & evaluation)
- `reports/prepare_data_summary.json` — số ảnh mỗi lớp/split, số ảnh skipped

---

## 🔧 Lớp 2 — Load & chuẩn hóa real-time (`src/data_loader.py`)

Áp dụng **mọi ảnh** ngay khi load vào `tf.data.Dataset` (cả train/val/test).

### Hàm chính

#### `_decode_image(path, label, img_size)` (dòng 49)

```python
img = tf.io.read_file(path)
img = tf.image.decode_image(img, channels=3, expand_animations=False)
img = tf.image.resize(img, [48, 48])         # IMG_SIZE từ config
img = tf.cast(img, tf.float32) / 255.0       # normalize [0, 1]
```

| Bước | Mục đích |
|---|---|
| `decode_image` | Đọc PNG/JPEG → tensor `uint8` |
| `resize` về `IMG_SIZE × IMG_SIZE` | Đồng nhất kích thước input cho model (GTSRB ảnh gốc 15-250px khác nhau) |
| Normalize `/ 255.0` | Đưa pixel về `[0, 1]` (giúp gradient ổn định, hội tụ nhanh hơn) |


#### `make_tf_dataset(filepaths, labels, ...)` (dòng 57)

Build `tf.data.Dataset` pipeline:

```python
ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
if shuffle: ds = ds.shuffle(buffer_size=2048, seed=SEED)
ds = ds.map(_decode_image, num_parallel_calls=AUTOTUNE)
if augment_fn: ds = ds.map(lambda x,y: (augment_fn(x), y), AUTOTUNE)
return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
```

Tham số `augment_fn` chỉ truyền vào cho train (xem Lớp 3).
`AUTOTUNE` cho phép TF tự song song hóa I/O & decode → tận dụng hết GPU.

#### `gather_filepaths(split_dir, class_folders)` (dòng 25)

Quét recursive `data/processed/<split>/<class>/`, gán nhãn theo index của
`class_folders` (đã sort). Hỗ trợ `.jpg`, `.jpeg`, `.png`, `.bmp`.

---

## 🔧 Lớp 3 — Data Augmentation (`src/preprocessing.py`)

Chỉ áp dụng cho **train set**, mỗi epoch sample random khác nhau → tăng
robustness, giảm overfitting.

### Hàm `augment(img)` (dòng 40)

Chuỗi 4 phép biến đổi áp dụng tuần tự lên 1 ảnh `[H, W, 3] ∈ [0, 1]`:

| # | Phép biến đổi | Tham số | Lý do với GTSRB |
|---|---|---|---|
| 1 | Affine: xoay + zoom + dịch chuyển | rot ±12°, zoom ±10%, trans ±6% | Camera trên xe nghiêng/rung, khoảng cách biển báo thay đổi |
| 2 | Random brightness | `±0.12` | Trời nắng vs bóng râm, sáng/tối |
| 3 | Random contrast | `0.85–1.15×` | Sương mù, ngược sáng |
| 4 | Random saturation | `0.85–1.15×` | Camera khác nhau, độ bão hòa khác |
| ❌ | **KHÔNG** flip ngang | — | Biển báo có hướng (rẽ trái ≠ rẽ phải) |

### Helper `_random_affine(img, max_deg, max_trans, max_zoom)` (dòng 7)

Implement xoay + zoom + dịch chuyển bằng `tf.raw_ops.ImageProjectiveTransformV3`
(thuần TF, hoạt động đúng trong `tf.data.Dataset.map`):

- Sinh ngẫu nhiên: `angle`, `zoom_factor`, `tx`, `ty`
- Tính ma trận affine 2×3 quanh tâm ảnh
- Apply transform với `BILINEAR` interpolation, `REFLECT` fill mode
  (tránh viền đen ở góc khi xoay)

### Hàm `preprocess_single_image(img, img_size)` (dòng 59)

Dùng cho **inference (1 ảnh)** trong `src/predict.py` và app Streamlit.
Chỉ resize + normalize, **không augment**:

```python
img = tf.image.resize(img, [img_size, img_size])
img = tf.cast(img, tf.float32) / 255.0
```

---

## ⚙️ Cấu hình liên quan (`src/config.py`)

| Hằng số | Giá trị | Ảnh hưởng đến lớp nào |
|---|---|---|
| `IMG_SIZE = 48` | 48 | Lớp 2 (resize), Lớp 3 (augment giữ nguyên size) |
| `IMG_CHANNELS = 3` | RGB | Lớp 2 (`decode_image(channels=3)`) |
| `BATCH_SIZE = 128` | 128 | Lớp 2 (`.batch()`) |
| `SEED = 42` | 42 | Lớp 1 (split), Lớp 2 (shuffle) — đảm bảo reproducible |

Trong `prepare_data.py`:

| Hằng số | Giá trị | Ý nghĩa |
|---|---|---|
| `VAL_RATIO_FROM_TRAIN = 0.10` | 10% | Tỉ lệ tách val từ train |
| `PADDING_RATIO = 0.05` | 5% | Padding khi crop ROI |
| `NUM_CLASSES = 43` | 43 | Số lớp GTSRB (assert chống lệch file) |

---

## 📁 Cấu trúc output sau tiền xử lý

```
data/
├── gtsrb_raw/                         # do prepare_data giải nén
│   ├── Train/{0..42}/*.png
│   ├── Test/*.png
│   ├── Train.csv, Test.csv, Meta.csv
│   └── Meta/*.png
└── processed/                         # do prepare_data sinh ra
    ├── train/
    │   ├── 00_Speed_limit_20km_h/*.png
    │   ├── 01_Speed_limit_30km_h/*.png
    │   └── ... (43 folder)
    ├── val/    (cấu trúc tương tự)
    └── test/   (cấu trúc tương tự)

models/
└── labels.json                        # 43 lớp metadata

reports/
└── prepare_data_summary.json          # số ảnh từng lớp × từng split
```

---

## ✅ Checklist verify sau khi chạy `prepare_data`

```bash
# 1. Số folder lớp ở mỗi split phải = 43
ls data/processed/train | wc -l        # → 43
ls data/processed/val | wc -l          # → 43
ls data/processed/test | wc -l         # → 43

# 2. Tổng số ảnh xấp xỉ
find data/processed/train -name "*.png" | wc -l    # ~35,288
find data/processed/val   -name "*.png" | wc -l    # ~3,921
find data/processed/test  -name "*.png" | wc -l    # ~12,630

# 3. labels.json đủ 43 entries
python -c "import json; print(len(json.load(open('models/labels.json'))))"
```

Hoặc xem chi tiết trong `reports/prepare_data_summary.json`.

---

## 🔍 Tham khảo thêm

- Code chính: `src/prepare_data.py`, `src/data_loader.py`, `src/preprocessing.py`
- Cấu hình: `src/config.py`
- Pipeline train sử dụng các lớp trên: `src/train.py` (hàm `main()`, dòng 63)
- Notebook chạy trên Colab: `notebooks/colab_train.ipynb` (mục 5)
