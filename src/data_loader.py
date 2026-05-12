"""Load dataset biển báo VN đã crop từ data/processed/{train,val,test}."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import config as C


def list_class_folders(split_dir: Path) -> list[str]:
    """Lấy danh sách folder lớp (đã sort) trong 1 split directory."""
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {split_dir}. Hãy chạy `python -m src.prepare_data` trước."
        )
    folders = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
    if not folders:
        raise FileNotFoundError(f"{split_dir} không có thư mục con nào.")
    return folders


def gather_filepaths(split_dir: Path, class_folders: list[str]):
    """Quét tất cả ảnh trong split_dir, gán nhãn theo index của class_folders."""
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
    """Trả về (train, val, test) dưới dạng (filepaths, labels) và class_folders."""
    class_folders = list_class_folders(C.PROCESSED_TRAIN_DIR)
    train = gather_filepaths(C.PROCESSED_TRAIN_DIR, class_folders)
    val = gather_filepaths(C.PROCESSED_VAL_DIR, class_folders)
    test = gather_filepaths(C.PROCESSED_TEST_DIR, class_folders)
    return train, val, test, class_folders


def _decode_image(path, label, img_size: int):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=C.IMG_CHANNELS, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def make_tf_dataset(filepaths, labels, num_classes: int,
                    batch_size: int = C.BATCH_SIZE,
                    img_size: int = C.IMG_SIZE,
                    shuffle: bool = False,
                    augment_fn=None) -> tf.data.Dataset:
    """Tạo tf.data.Dataset từ list filepaths + labels."""
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(filepaths), 2048), seed=C.SEED)
    ds = ds.map(lambda p, y: _decode_image(p, y, img_size),
                num_parallel_calls=tf.data.AUTOTUNE)
    if augment_fn is not None:
        ds = ds.map(lambda x, y: (augment_fn(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_class_metadata() -> list[dict]:
    """Đọc labels.json (do prepare_data.py sinh ra)."""
    if not C.LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {C.LABELS_PATH}. Hãy chạy `python -m src.prepare_data` trước."
        )
    with open(C.LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_display_names(class_folders: list[str], lang: str = "vi") -> list[str]:
    """Trả về tên hiển thị (tiếng Việt mặc định) theo thứ tự class_folders."""
    meta = load_class_metadata()
    folder_map = {m["folder"]: m for m in meta}
    key = "name_vi" if lang == "vi" else "name_en"
    return [folder_map.get(f, {}).get(key, f) for f in class_folders]


def load_labels(lang: str = "vi") -> list[str]:
    """Trả về list tên hiển thị theo thứ tự lớp của model.

    Nguồn ưu tiên: `models/labels.json` (do prepare_data tạo ra) — không phụ thuộc
    vào `data/processed/train/` còn tồn tại ở local hay không.

    Thứ tự lớp = sort theo `folder` (folder bắt đầu bằng idx 2 chữ số nên
    sort theo folder tương đương sort theo idx, đúng với thứ tự khi train).
    """
    if C.LABELS_PATH.exists():
        meta = load_class_metadata()
        meta_sorted = sorted(meta, key=lambda m: m["folder"])
        key = "name_vi" if lang == "vi" else "name_en"
        return [m.get(key, m["folder"]) for m in meta_sorted]
    class_folders = list_class_folders(C.PROCESSED_TRAIN_DIR)
    return get_display_names(class_folders, lang=lang)
