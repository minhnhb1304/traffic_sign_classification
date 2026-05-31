"""Streamlit demo: phân loại biển báo giao thông GTSRB.

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
# Layout "centered" cho col1 (tỷ lệ 2:1) rộng ~460px → giới hạn cạnh dài < 440
# để ảnh không tràn sang col2 (gây che kết quả dự đoán).
CROPPER_MIN_SIDE = 320   # upscale ảnh nhỏ để có vùng thao tác đủ rộng
CROPPER_MAX_SIDE = 420   # downscale ảnh lớn để vừa khít col1, không tràn layout


st.set_page_config(page_title="Phân loại biển báo giao thông",
                   page_icon="🚦", layout="centered")


def _fit_for_cropper(img: Image.Image,
                     min_side: int = CROPPER_MIN_SIDE,
                     max_side: int = CROPPER_MAX_SIDE) -> Image.Image:
    """Resize ảnh để cropper UI vừa khít vùng thao tác.

    Model luôn resize ROI về IMG_SIZE nên scale ở đây không ảnh hưởng accuracy.
    """
    w, h = img.size
    short, long_ = min(w, h), max(w, h)
    if short < min_side:
        scale = min_side / short
    elif long_ > max_side:
        scale = max_side / long_
    else:
        return img
    new_size = (round(w * scale), round(h * scale))
    return img.resize(new_size, Image.LANCZOS)


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


def _render_crop_and_predict(model, labels, image: Image.Image,
                             cropper_key: str) -> None:
    """Phần UI chung: crop ROI (tuỳ chọn) + hiển thị kết quả dự đoán."""
    top_k = st.sidebar.slider("Số kết quả hiển thị", 1, 5, 3)
    use_crop = st.sidebar.checkbox(
        "✂️ Crop ROI thủ công", value=True,
        help="Kéo khung xanh để chọn vùng chứa biển báo. "
             "Bỏ tick để predict trực tiếp ảnh full-size.",
    )
    aspect_choice = st.sidebar.radio(
        "Tỷ lệ khung crop", ["Vuông 1:1", "Tự do"],
        index=0, disabled=not use_crop,
        help="Biển báo thường vuông → giữ 1:1 cho ổn định.",
    )
    aspect_ratio = (1, 1) if aspect_choice == "Vuông 1:1" else None

    if not use_crop:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Ảnh đầu vào (full-size, không crop)",
                     use_container_width=True)
        with col2:
            _show_results(model, labels, image, top_k)
            st.warning("⚠️ Mode full-size: Ảnh không crop → accuracy giảm đáng kể.")
        return

    display_image = _fit_for_cropper(image)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("🖱️ Kéo khung xanh để chọn vùng biển báo ")
        cropped = st_cropper(
            display_image, realtime_update=True, box_color="#00FF00",
            aspect_ratio=aspect_ratio, return_type="image", key=cropper_key,
        )
    with col2:
        if cropped is None or min(cropped.size) < MIN_CROP_PX:
            st.warning(f"Khung crop quá nhỏ (<{MIN_CROP_PX}px). "
                       "Kéo khung lớn hơn để predict.")
            return
        st.image(cropped, use_container_width=True)
        _show_results(model, labels, cropped, top_k)


def render_upload_tab(model, labels):
    uploaded = st.file_uploader(
        "Tải lên ảnh biển báo (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded is None:
        st.info("Hãy tải lên một ảnh biển báo để bắt đầu.")
        return

    image = Image.open(uploaded).convert("RGB")
    _render_crop_and_predict(model, labels, image, cropper_key="roi_cropper")


def render_camera_tab(model, labels):
    st.caption("Bấm nút bên dưới để bật camera, rồi chụp 1 ảnh biển báo.")
    snapshot = st.camera_input("Chụp ảnh biển báo")
    if snapshot is None:
        st.info("Hãy chụp một ảnh để bắt đầu dự đoán.")
        return

    image = Image.open(snapshot).convert("RGB")
    _render_crop_and_predict(model, labels, image, cropper_key="camera_cropper")


def main():
    st.title("🚦 Phân loại biển báo giao thông")

    if not C.MODEL_PATH.exists():
        st.error(
            f"Chưa tìm thấy model tại {C.MODEL_PATH}. "
            "Hãy chạy `python -m src.train` trước."
        )
        return

    model, labels = load_model_and_labels()

    mode = st.sidebar.radio(
        "Chế độ",
        ["📁 Upload ảnh", "📷 Chụp ảnh", "🎥 Camera realtime"],
    )

    if mode == "🎥 Camera realtime":
        render_realtime_tab(model, labels)
    elif mode == "📷 Chụp ảnh":
        render_camera_tab(model, labels)
    else:
        render_upload_tab(model, labels)


if __name__ == "__main__":
    main()
