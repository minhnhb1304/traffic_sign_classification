"""Wrap model + labels dùng chung cho cả mode upload và realtime.

Tách khỏi `streamlit_app.py` để VideoProcessor có thể import mà không kéo theo
phụ thuộc Streamlit nặng.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from src import config as C
from src.data_loader import load_labels
from src.preprocessing import preprocess_single_image


def load_model_and_labels():
    """Load Keras model + labels (gọi 1 lần, nên cache bằng @st.cache_resource)."""
    model = tf.keras.models.load_model(C.MODEL_PATH)
    labels = load_labels()
    return model, labels


def predict_array(model, image_rgb: np.ndarray, top_k: int = 3) -> list[tuple[int, float]]:
    """Dự đoán cho 1 ảnh RGB (HxWx3, uint8 hoặc float).

    Trả về list[(class_idx, prob)] đã sort giảm dần, length = top_k.
    """
    tensor = tf.convert_to_tensor(image_rgb)
    tensor = preprocess_single_image(tensor, C.IMG_SIZE)
    tensor = tf.expand_dims(tensor, axis=0)
    probs = model.predict(tensor, verbose=0)[0]
    idx = np.argsort(probs)[::-1][:top_k]
    return [(int(i), float(probs[i])) for i in idx]


def predict_proba(model, image_rgb: np.ndarray) -> np.ndarray:
    """Trả về full softmax vector (dùng cho smoothing trong realtime)."""
    tensor = tf.convert_to_tensor(image_rgb)
    tensor = preprocess_single_image(tensor, C.IMG_SIZE)
    tensor = tf.expand_dims(tensor, axis=0)
    return model.predict(tensor, verbose=0)[0]
