"""Inference cho 1 ảnh đơn lẻ.

Cách dùng:
    python -m src.predict --image path/to/sign.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import config as C
from .data_loader import load_labels
from .preprocessing import preprocess_single_image


def predict_image(image_path: str | Path, model_type: str = "gtsrb", top_k: int = 3):
    """Dự đoán nhãn cho 1 ảnh, trả về list (class_name, prob) top-k."""
    model_path = C.MODEL_PATH_VN if model_type.lower() in ("vn", "vietnam") else C.MODEL_PATH
    model = tf.keras.models.load_model(model_path)
    class_names = load_labels(model_type=model_type)

    img_raw = tf.io.read_file(str(image_path))
    img = tf.image.decode_image(img_raw, channels=C.IMG_CHANNELS, expand_animations=False)
    img = preprocess_single_image(img, C.IMG_SIZE)
    img = tf.expand_dims(img, axis=0)

    probs = model.predict(img, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in top_idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Đường dẫn ảnh cần dự đoán")
    parser.add_argument("--model-type", default="gtsrb", choices=["gtsrb", "vn"],
                        help="Loại model: 'gtsrb' (43 lớp) hoặc 'vn' (biển báo VN)")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    results = predict_image(args.image, model_type=args.model_type, top_k=args.top_k)
    print(f"\nKết quả dự đoán ({args.model_type.upper()}) cho: {args.image}")
    for rank, (name, prob) in enumerate(results, start=1):
        print(f"  Top-{rank}: {name:30s}  {prob*100:6.2f}%")


if __name__ == "__main__":
    main()

