"""Streamlit demo: nhận diện biển báo giao thông GTSRB.

Chạy:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_cropper import st_cropper

# Thêm root vào sys.path để import được package src
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src.data_loader import load_labels  # noqa: E402
from src.preprocessing import preprocess_single_image  # noqa: E402
from app.realtime.tab import render_realtime_tab  # noqa: E402

MIN_CROP_PX = 16


st.set_page_config(page_title="Nhận diện biển báo GTSRB",
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


def _show_results(model, labels, pil_image: Image.Image, top_k: int) -> None:
    with st.spinner("Đang dự đoán..."):
        results = predict(model, labels, pil_image, top_k=top_k)
    st.subheader("Kết quả")
    for rank, (name, prob) in enumerate(results, start=1):
        st.write(f"**Top-{rank}:** {name}")
        st.progress(prob, text=f"{prob*100:.2f}%")


def render_upload_tab(model, labels):
    top_k = st.sidebar.slider("Số kết quả top-k", 1, 5, 3)
    use_crop = st.sidebar.checkbox(
        "✂️ Crop ROI thủ công (khuyến nghị)", value=True,
        help="Kéo khung xanh để chọn vùng chứa biển báo. "
             "Bỏ tick để predict trực tiếp ảnh full-size (acc giảm mạnh).",
    )
    aspect_choice = st.sidebar.radio(
        "Tỷ lệ khung crop", ["Vuông 1:1", "Tự do"],
        index=0, disabled=not use_crop,
        help="Biển báo thường vuông → giữ 1:1 cho ổn định.",
    )
    aspect_ratio = (1, 1) if aspect_choice == "Vuông 1:1" else None

    uploaded = st.file_uploader(
        "Tải lên ảnh biển báo (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded is None:
        st.info("Hãy tải lên một ảnh biển báo để bắt đầu.")
        return

    image = Image.open(uploaded).convert("RGB")

    if not use_crop:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Ảnh đầu vào (full-size, không crop)",
                     use_container_width=True)
        with col2:
            st.warning("⚠️ Mode full-size: model là **classifier**, "
                       "ảnh không crop → accuracy giảm đáng kể.")
            _show_results(model, labels, image, top_k)
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**🖱️ Kéo khung xanh để chọn vùng biển báo (ROI)**")
        cropped = st_cropper(
            image, realtime_update=True, box_color="#00FF00",
            aspect_ratio=aspect_ratio, return_type="image", key="roi_cropper",
        )
    with col2:
        if cropped is None or min(cropped.size) < MIN_CROP_PX:
            st.warning(f"Khung crop quá nhỏ (<{MIN_CROP_PX}px). "
                       "Kéo khung lớn hơn để predict.")
            return
        st.markdown(f"**ROI:** `{cropped.size[0]}×{cropped.size[1]}` px "
                    f"→ resize `{C.IMG_SIZE}×{C.IMG_SIZE}`")
        st.image(cropped, use_container_width=True)
        _show_results(model, labels, cropped, top_k)


def main():
    st.title("🚦 Nhận diện biển báo giao thông GTSRB")
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
    mode = st.sidebar.radio("Chế độ", ["📁 Upload ảnh", "🎥 Camera realtime"])

    if mode == "🎥 Camera realtime":
        render_realtime_tab(model, labels)
    else:
        render_upload_tab(model, labels)


if __name__ == "__main__":
    main()
