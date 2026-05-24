"""UI cho mode 'Camera realtime' — render trong main thread của Streamlit."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from src import config as C
from .video_processor import SignVideoProcessor

SNAPSHOTS_DIR = C.REPORTS_DIR / "snapshots"

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

MEDIA_CONSTRAINTS = {"video": {"width": {"ideal": 640}}, "audio": False}


def _save_snapshot(frame_bgr, label: str, conf: float) -> Path:
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() else "_" for ch in label)[:40] or "snapshot"
    out = SNAPSHOTS_DIR / f"{ts}_{safe}_{int(conf*100):03d}.png"
    cv2.imwrite(str(out), frame_bgr)
    return out


def render_realtime_tab(model, labels: list[str]) -> None:
    st.subheader("🎥 Phân loại biển báo bằng camera")
    st.caption("Đưa biển báo vào khung vuông xanh ở giữa hình.")

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Tham số realtime**")
        roi_ratio = st.slider(
            "Kích thước ROI",
            0.30, 0.90, 0.50, 0.05,
            help="Tỷ lệ vùng vuông ở giữa khung hình so với cạnh ngắn của frame. "
                 "Tăng giá trị nếu biển báo ở gần camera, giảm nếu biển báo ở xa.",
        )
        threshold = st.slider(
            "Ngưỡng confidence",
            0.30, 0.95, 0.60, 0.05,
            help="Mức xác suất tối thiểu để hiển thị nhãn dự đoán. "
                 "Dưới ngưỡng này hệ thống coi như không phát hiện biển báo.",
        )
        smooth_window = 8

    ctx = webrtc_streamer(
        key="sign-realtime",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints=MEDIA_CONSTRAINTS,
        video_processor_factory=SignVideoProcessor,
        async_processing=True,
    )

    if ctx.video_processor is not None:
        ctx.video_processor.configure(
            model=model, labels=labels,
            roi_ratio=roi_ratio, threshold=threshold,
            smooth_window=smooth_window, show_fps=0
        )

    col1, col2 = st.columns([1, 3])
    with col1:
        snap_btn = st.button("📸 Chụp snapshot",
                             disabled=(ctx.video_processor is None),
                             use_container_width=True)
    with col2:
        if ctx.state.playing:
            st.success("🟢 Camera đang chạy")
        elif ctx.video_processor is None:
            st.info("🟡 Bấm **START** trên khung video để bật camera")
        else:
            st.warning("🔴 Camera đang tắt — bấm **START** để chạy lại")

    if snap_btn and ctx.video_processor is not None:
        frame, label, conf = ctx.video_processor.snapshot()
        if frame is None:
            st.warning("Chưa có frame nào — hãy chờ camera chạy trước.")
        else:
            saved = _save_snapshot(frame, label, conf)
            st.success(f"Đã lưu snapshot: `{saved.relative_to(C.ROOT_DIR)}`")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                     caption=f"{label} ({conf*100:.1f}%)",
                     use_container_width=True)

    with st.expander("Lưu ý"):
        st.markdown(
            "- Mỗi frame chỉ phân loại 1 biển báo trong khung ROI ở giữa. \n "
            "- Ảnh snapshot lưu tại `reports/snapshots/`."
        )
