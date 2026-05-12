"""Chẩn đoán nhanh model: kiểm tra weights, output distribution, accuracy nhanh."""
from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
from PIL import Image

from src import config as C
from src.data_loader import list_class_folders, get_display_names


def load_first_n_per_class(split_dir: Path, class_folders, n: int = 2):
    paths, true_idx = [], []
    for i, cname in enumerate(class_folders):
        cdir = split_dir / cname
        if not cdir.exists():
            continue
        imgs = sorted(cdir.glob("*.jpg"))[:n]
        for p in imgs:
            paths.append(p)
            true_idx.append(i)
    return paths, true_idx


def to_input(p: Path) -> np.ndarray:
    img = Image.open(p).convert("RGB").resize((C.IMG_SIZE, C.IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def main():
    print(f">>> Loading model: {C.MODEL_PATH}")
    model = tf.keras.models.load_model(C.MODEL_PATH)
    print(f"    Output shape: {model.output_shape}")
    print(f"    Total params: {model.count_params():,}")

    # Kiểm tra weights có được train hay không (variance khác 0 đáng kể)
    last_layer = model.get_layer("predictions")
    w = last_layer.get_weights()[0]
    print(f"    Last layer weight stats: mean={w.mean():+.4f}, "
          f"std={w.std():.4f}, min={w.min():+.4f}, max={w.max():+.4f}")

    class_folders = list_class_folders(C.PROCESSED_TRAIN_DIR)
    names_vi = get_display_names(class_folders, lang="vi")
    print(f"    Số lớp local: {len(class_folders)}")

    # Lấy 2 ảnh / class từ train
    print("\n>>> Predict trên 2 ảnh/class từ TRAIN set...")
    paths, true_idx = load_first_n_per_class(C.PROCESSED_TRAIN_DIR, class_folders, n=2)
    batch = np.stack([to_input(p) for p in paths], axis=0)
    probs = model.predict(batch, batch_size=32, verbose=0)
    pred_idx = probs.argmax(axis=1)
    top1 = np.mean(pred_idx == np.array(true_idx)) * 100
    top1_max = probs.max(axis=1).mean() * 100
    print(f"    TRAIN top-1 accuracy: {top1:.2f}%  ({sum(pred_idx == np.array(true_idx))}/{len(paths)})")
    print(f"    TRAIN avg top-1 confidence: {top1_max:.2f}%")

    # Same với test
    print("\n>>> Predict trên 2 ảnh/class từ TEST set...")
    paths_t, true_t = load_first_n_per_class(C.PROCESSED_TEST_DIR, class_folders, n=2)
    if paths_t:
        batch_t = np.stack([to_input(p) for p in paths_t], axis=0)
        probs_t = model.predict(batch_t, batch_size=32, verbose=0)
        pred_t = probs_t.argmax(axis=1)
        top1_t = np.mean(pred_t == np.array(true_t)) * 100
        top1_max_t = probs_t.max(axis=1).mean() * 100
        print(f"    TEST  top-1 accuracy: {top1_t:.2f}%  ({sum(pred_t == np.array(true_t))}/{len(paths_t)})")
        print(f"    TEST  avg top-1 confidence: {top1_max_t:.2f}%")

    # 5 mẫu đầu — show prediction
    print("\n>>> Chi tiết 5 ảnh đầu của TRAIN:")
    for i in range(min(5, len(paths))):
        true_name = names_vi[true_idx[i]]
        pred_name = names_vi[pred_idx[i]]
        ok = "✅" if pred_idx[i] == true_idx[i] else "❌"
        print(f"  {ok} {paths[i].parent.name:18s} | true='{true_name}'  ->  pred='{pred_name}' ({probs[i].max()*100:.1f}%)")


if __name__ == "__main__":
    main()
