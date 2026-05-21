"""Vẽ ROI box + label lên frame BGR (in-place không phải lựa chọn — trả ảnh mới)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def _bgr_to_rgb(color: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = color
    return r, g, b


@lru_cache(maxsize=1)
def _load_unicode_font(size: int = 22) -> ImageFont.ImageFont:
    """Load font TrueType hỗ trợ Unicode để vẽ tiếng Việt lên frame realtime."""
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _draw_unicode_label(out_bgr: np.ndarray, text: str, x: int, y: int,
                        bg_color: tuple[int, int, int]) -> np.ndarray:
    """Vẽ label Unicode bằng Pillow rồi trả ảnh BGR cho OpenCV/WebRTC."""
    font = _load_unicode_font()
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(out_rgb)
    draw = ImageDraw.Draw(pil_img)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    bg_rgb = _bgr_to_rgb(bg_color)
    fg_rgb = _bgr_to_rgb(BLACK)
    x1 = min(x + tw + 8, out_bgr.shape[1] - 1)
    y1 = min(y + th + 8, out_bgr.shape[0] - 1)
    draw.rectangle((x, y, x1, y1), fill=bg_rgb)
    draw.text((x + 4, y + 2), text, font=font, fill=fg_rgb)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


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
    label_y = max(y0 - 36, 0)
    out = _draw_unicode_label(out, text, x0, label_y, color)

    if fps is not None:
        h, _ = out.shape[:2]
        cv2.putText(out, f"FPS: {fps:.1f}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)

    return out
