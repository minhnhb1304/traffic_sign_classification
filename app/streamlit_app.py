"""Streamlit demo: nhận diện biển báo giao thông Việt Nam.

Chạy:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Thêm root vào sys.path để import được package src
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src.data_loader import load_labels  # noqa: E402
from src.preprocessing import preprocess_single_image  # noqa: E402


st.set_page_config(page_title="Nhận diện biển báo VN",
                   page_icon="🚦", layout="centered")


@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(C.MODEL_PATH)
    labels = load_labels()
    return model, labels


def predict(model, labels, pil_image: Image.Image, top_k: int = 3):
    arr = np.array(pil_image.convert("RGB"))
    tensor = tf.convert_to_tensor(arr)
    tensor = preprocess_single_image(tensor, C.IMG_SIZE)
    tensor = tf.expand_dims(tensor, axis=0)
    probs = model.predict(tensor, verbose=0)[0]
    idx = np.argsort(probs)[::-1][:top_k]
    return [(labels[i], float(probs[i])) for i in idx]


def main():
    st.title("🚦 Nhận diện biển báo giao thông Việt Nam")
    st.caption("Demo CNN tự xây dựng — Đồ án cuối kỳ")

    if not C.MODEL_PATH.exists():
        st.error(
            f"Chưa tìm thấy model tại {C.MODEL_PATH}. "
            "Hãy chạy `python -m src.train` trước."
        )
        return

    model, labels = load_model_and_labels()
    st.sidebar.write(f"**Số lớp:** {len(labels)}")
    st.sidebar.write(f"**Kích thước ảnh:** {C.IMG_SIZE}×{C.IMG_SIZE}")
    top_k = st.sidebar.slider("Số kết quả top-k", 1, 5, 3)

    uploaded = st.file_uploader(
        "Tải lên ảnh biển báo (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded is None:
        st.info("Hãy tải lên một ảnh biển báo để bắt đầu.")
        return

    image = Image.open(uploaded)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Ảnh đầu vào", use_container_width=True)

    with col2:
        with st.spinner("Đang dự đoán..."):
            results = predict(model, labels, image, top_k=top_k)
        st.subheader("Kết quả")
        for rank, (name, prob) in enumerate(results, start=1):
            st.write(f"**Top-{rank}:** {name}")
            st.progress(prob, text=f"{prob*100:.2f}%")


if __name__ == "__main__":
    main()
