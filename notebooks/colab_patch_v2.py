# =====================================================================
# Patch v3 — Fix collapse: IMG_SIZE=64, MIN_SAMPLES=30, augment bằng tf.image,
#           cap class_weight ở 5x.
# Paste TOÀN BỘ cell này vào Colab và Run.
# =====================================================================
from pathlib import Path

FILES = {}

FILES["src/config.py"] = '''"""Cấu hình toàn bộ dự án: đường dẫn, hyperparameters."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

RAW_IMAGES_DIR = DATA_DIR / "images"
RAW_LABELS_DIR = DATA_DIR / "labels"
RAW_SPLIT_DIR = DATA_DIR / "split_dataset"
CLASSES_FILE = DATA_DIR / "classes.txt"
CLASSES_EN_FILE = DATA_DIR / "classes_en.txt"
CLASSES_VIE_FILE = DATA_DIR / "classes_vie.txt"

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_TRAIN_DIR = PROCESSED_DIR / "train"
PROCESSED_VAL_DIR = PROCESSED_DIR / "val"
PROCESSED_TEST_DIR = PROCESSED_DIR / "test"

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 64
IMG_CHANNELS = 3
SEED = 42

BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 5e-4
EARLY_STOPPING_PATIENCE = 15
LR_REDUCE_PATIENCE = 5

MODEL_NAME = "custom_cnn_v1"
MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.keras"
LABELS_PATH = MODELS_DIR / "labels.json"
'''

FILES["src/preprocessing.py"] = '''"""Tiền xử lý ảnh và data augmentation cho biển báo VN."""
import math

import tensorflow as tf


def _random_affine(img, max_deg=10.0, max_trans=0.05, max_zoom=0.08):
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    angle = tf.random.uniform([], -max_deg, max_deg) * math.pi / 180.0
    cos_a, sin_a = tf.cos(angle), tf.sin(angle)
    zoom = 1.0 + tf.random.uniform([], -max_zoom, max_zoom)
    tx = tf.random.uniform([], -max_trans, max_trans) * w
    ty = tf.random.uniform([], -max_trans, max_trans) * h
    cx, cy = w / 2.0, h / 2.0
    a0, a1 = cos_a / zoom, sin_a / zoom
    b0, b1 = -sin_a / zoom, cos_a / zoom
    a2 = cx - cx * a0 - cy * a1 + tx
    b2 = cy - cx * b0 - cy * b1 + ty
    transforms = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[tf.newaxis, :]
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...], transforms=transforms,
        output_shape=tf.shape(img)[:2], fill_value=0.0,
        interpolation="BILINEAR", fill_mode="REFLECT",
    )
    return out[0]


def augment(img):
    img = _random_affine(img, max_deg=12.0, max_trans=0.06, max_zoom=0.10)
    img = tf.image.random_brightness(img, max_delta=0.12)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    img = tf.image.random_saturation(img, lower=0.85, upper=1.15)
    return tf.clip_by_value(img, 0.0, 1.0)


def build_augmentation_pipeline():
    return augment


def preprocess_single_image(img, img_size):
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0
    return img
'''

FILES["src/model.py"] = '''"""Kiến trúc CNN tự xây."""
import tensorflow as tf
from tensorflow.keras import Model, layers

from . import config as C


def build_cnn(num_classes, img_size=C.IMG_SIZE, channels=C.IMG_CHANNELS):
    inputs = layers.Input(shape=(img_size, img_size, channels), name="image_input")
    x = _conv_block(inputs, filters=32, dropout=0.20, name="block1")
    x = _conv_block(x, filters=64, dropout=0.25, name="block2")
    x = _conv_block(x, filters=128, dropout=0.30, name="block3")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Dropout(0.5, name="fc1_drop")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return Model(inputs=inputs, outputs=outputs, name=C.MODEL_NAME)


def _conv_block(x, filters, dropout, name):
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    x = layers.MaxPooling2D(pool_size=2, name=f"{name}_pool")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    return x


def compile_model(model, learning_rate=C.LEARNING_RATE):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")],
    )
    return model
'''

FILES["src/data_loader.py"] = '''"""Load dataset đã crop từ data/processed/{train,val,test}."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import config as C


def list_class_folders(split_dir):
    if not split_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy {split_dir}.")
    folders = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
    if not folders:
        raise FileNotFoundError(f"{split_dir} rỗng.")
    return folders


def gather_filepaths(split_dir, class_folders):
    folder_to_idx = {name: i for i, name in enumerate(class_folders)}
    filepaths, labels = [], []
    for cname in class_folders:
        cdir = split_dir / cname
        if not cdir.exists():
            continue
        for img_path in cdir.rglob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                filepaths.append(str(img_path))
                labels.append(folder_to_idx[cname])
    return np.array(filepaths), np.array(labels)


def load_train_val_test():
    class_folders = list_class_folders(C.PROCESSED_TRAIN_DIR)
    train = gather_filepaths(C.PROCESSED_TRAIN_DIR, class_folders)
    val = gather_filepaths(C.PROCESSED_VAL_DIR, class_folders)
    test = gather_filepaths(C.PROCESSED_TEST_DIR, class_folders)
    return train, val, test, class_folders


def _decode_image(path, label, img_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=C.IMG_CHANNELS, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def make_tf_dataset(filepaths, labels, num_classes,
                    batch_size=C.BATCH_SIZE, img_size=C.IMG_SIZE,
                    shuffle=False, augment_fn=None):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(filepaths), 2048), seed=C.SEED)
    ds = ds.map(lambda p, y: _decode_image(p, y, img_size),
                num_parallel_calls=tf.data.AUTOTUNE)
    if augment_fn is not None:
        ds = ds.map(lambda x, y: (augment_fn(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_class_metadata():
    if not C.LABELS_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy {C.LABELS_PATH}.")
    with open(C.LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_display_names(class_folders, lang="vi"):
    meta = load_class_metadata()
    folder_map = {m["folder"]: m for m in meta}
    key = "name_vi" if lang == "vi" else "name_en"
    return [folder_map.get(f, {}).get(key, f) for f in class_folders]


def load_labels():
    class_folders = list_class_folders(C.PROCESSED_TRAIN_DIR)
    return get_display_names(class_folders, lang="vi")
'''

FILES["src/train.py"] = '''"""Script huấn luyện."""
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from . import config as C
from .data_loader import load_train_val_test, make_tf_dataset
from .model import build_cnn, compile_model
from .preprocessing import build_augmentation_pipeline


def get_callbacks():
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(C.MODEL_PATH), save_best_only=True,
            monitor="val_loss", mode="min", verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min",
            patience=C.EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=C.LR_REDUCE_PATIENCE, min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(C.REPORTS_DIR / "training_log.csv")),
    ]


def compute_class_weights(y_train, num_classes, max_weight=5.0):
    counts = Counter(int(y) for y in y_train)
    total = sum(counts.values())
    raw = {i: total / (num_classes * counts.get(i, 1)) for i in range(num_classes)}
    return {i: min(w, max_weight) for i, w in raw.items()}


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train"); axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history.history["accuracy"], label="train"); axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def main():
    print(">>> Bước 1: Load dataset...")
    (x_tr, y_tr), (x_v, y_v), (x_te, y_te), class_folders = load_train_val_test()
    num_classes = len(class_folders)
    print(f"    Số lớp: {num_classes} | Train: {len(x_tr)} | Val: {len(x_v)} | Test: {len(x_te)}")

    print(">>> Bước 2: Tạo tf.data.Dataset...")
    aug = build_augmentation_pipeline()
    train_ds = make_tf_dataset(x_tr, y_tr, num_classes, shuffle=True, augment_fn=aug)
    val_ds = make_tf_dataset(x_v, y_v, num_classes)
    test_ds = make_tf_dataset(x_te, y_te, num_classes)

    print(">>> Bước 3: Build & compile model...")
    model = compile_model(build_cnn(num_classes=num_classes))
    model.summary()

    print(">>> Bước 4: Huấn luyện...")
    cw = compute_class_weights(y_tr, num_classes)
    print(f"    Class weights: min={min(cw.values()):.2f}, max={max(cw.values()):.2f}")
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=C.EPOCHS, callbacks=get_callbacks(),
                        class_weight=cw, verbose=1)

    print(">>> Bước 5: Đánh giá test...")
    test_metrics = model.evaluate(test_ds, return_dict=True, verbose=1)
    print(f"    Test metrics: {test_metrics}")
    with open(C.REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    plot_history(history, C.FIGURES_DIR / "training_curves.png")
    print(f">>> Hoàn tất. Model: {C.MODEL_PATH}")


if __name__ == "__main__":
    main()
'''

print(f">>> Sẽ ghi đè {len(FILES)} file:")
for path, content in FILES.items():
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding="utf-8")
    print(f"  ✓ {path} ({len(content)} bytes)")

# Cập nhật MIN_SAMPLES_PER_CLASS trong prepare_data.py (5 -> 30)
import re
pd_path = Path("src/prepare_data.py")
if pd_path.exists():
    txt = pd_path.read_text(encoding="utf-8")
    new_txt = re.sub(r"MIN_SAMPLES_PER_CLASS\s*=\s*\d+",
                     "MIN_SAMPLES_PER_CLASS = 30", txt)
    if new_txt != txt:
        pd_path.write_text(new_txt, encoding="utf-8")
        print("  ✓ src/prepare_data.py: MIN_SAMPLES_PER_CLASS -> 30")

# Xóa model cũ, log cũ và data/processed/ cũ (cần re-prepare với threshold mới)
import os, shutil
for f in ["models/custom_cnn_v1.keras", "models/labels.json",
          "reports/training_log.csv", "reports/metrics.json"]:
    if os.path.exists(f):
        os.remove(f); print(f"  🗑 Removed old {f}")
if os.path.isdir("data/processed"):
    shutil.rmtree("data/processed")
    print("  🗑 Removed data/processed/ (cần re-prepare với threshold mới)")
print("\n✅ Patch applied. Tiếp theo chạy: prepare_data → train.")
