"""SignVideoProcessor — chạy trong worker thread của streamlit-webrtc.

Mỗi frame:
  1. Convert BGR → RGB ROI center crop
  2. Predict softmax bằng model Keras (đã cache ở main thread)
  3. Smooth N frame gần nhất (rolling mean)
  4. Vẽ overlay (xanh nếu vượt threshold, đỏ nếu không)
  5. Push frame + (label, conf) vào shared state để main thread snapshot
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from .inference import predict_proba
from .overlay import compute_center_roi, draw_overlay


class SignVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.labels: list[str] = []
        self.roi_ratio: float = 0.5
        self.threshold: float = 0.6
        self.smooth_window: int = 5
        self.show_fps: bool = False

        self._buf: deque[np.ndarray] = deque(maxlen=self.smooth_window)
        self._lock = threading.Lock()
        self._last_frame_bgr: Optional[np.ndarray] = None
        self._last_label: str = ""
        self._last_conf: float = 0.0
        self._frame_count = 0
        self._t0 = time.time()

    def configure(self, *, model, labels, roi_ratio, threshold, smooth_window, show_fps):
        with self._lock:
            self.model = model
            self.labels = labels
            self.roi_ratio = float(roi_ratio)
            self.threshold = float(threshold)
            if smooth_window != self.smooth_window:
                self.smooth_window = int(smooth_window)
                self._buf = deque(maxlen=self.smooth_window)
            self.show_fps = bool(show_fps)

    def snapshot(self) -> tuple[Optional[np.ndarray], str, float]:
        with self._lock:
            frame = None if self._last_frame_bgr is None else self._last_frame_bgr.copy()
            return frame, self._last_label, self._last_conf

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if self.model is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        h, w = img.shape[:2]
        x0, y0, side = compute_center_roi(h, w, self.roi_ratio)
        roi_bgr = img[y0:y0 + side, x0:x0 + side]
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

        try:
            probs = predict_proba(self.model, roi_rgb)
        except Exception:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        self._buf.append(probs)
        avg = np.mean(self._buf, axis=0)
        idx, conf = int(np.argmax(avg)), float(np.max(avg))
        detected = conf >= self.threshold
        label = self.labels[idx] if detected else "— không phát hiện —"

        fps = None
        if self.show_fps:
            self._frame_count += 1
            elapsed = time.time() - self._t0
            if elapsed >= 1.0:
                fps = self._frame_count / elapsed
                self._frame_count = 0
                self._t0 = time.time()

        annotated = draw_overlay(img, (x0, y0, side), label, conf,
                                 detected=detected, fps=fps)

        with self._lock:
            self._last_frame_bgr = annotated
            self._last_label = label
            self._last_conf = conf

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")
