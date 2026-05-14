"""Vẽ ROI box + label lên frame BGR (in-place không phải lựa chọn — trả ảnh mới)."""
from __future__ import annotations

import cv2
import numpy as np

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def compute_center_roi(h: int, w: int, ratio: float) -> tuple[int, int, int]:
    """Trả về (x0, y0, side) — vùng vuông giữa frame, cạnh = min(h, w) * ratio."""
    side = int(min(h, w) * ratio)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return x0, y0, side


def draw_overlay(img_bgr: np.ndarray, roi: tuple[int, int, int],
                 label: str, conf: float, *,
                 detected: bool = True,
                 fps: float | None = None) -> np.ndarray:
    """Vẽ ROI box + nhãn + confidence lên ảnh BGR. Trả về ảnh mới (không sửa input)."""
    out = img_bgr.copy()
    x0, y0, side = roi
    color = GREEN if detected else RED

    cv2.rectangle(out, (x0, y0), (x0 + side, y0 + side), color, 2)

    text = f"{label}  {conf*100:.1f}%" if detected else label
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    label_y = max(y0 - 8, th + 8)
    cv2.rectangle(out, (x0, label_y - th - 6),
                  (x0 + tw + 8, label_y + baseline - 2), color, -1)
    cv2.putText(out, text, (x0 + 4, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2, cv2.LINE_AA)

    if fps is not None:
        h, _ = out.shape[:2]
        cv2.putText(out, f"FPS: {fps:.1f}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)

    return out
