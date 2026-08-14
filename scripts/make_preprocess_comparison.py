"""Tạo hình so sánh ảnh TRƯỚC và SAU bước tiền xử lý (resize 48x48 + normalize).

Chạy:  python scripts/make_preprocess_comparison.py [đường_dẫn_ảnh]
Mặc định dùng demo_images/tier1_gtsrb/03_speed_30.png.
Kết quả lưu tại reports/figures/preprocessing_comparison.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402


def main():
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "demo_images" / "tier1_gtsrb" / "03_speed_30.png")
    if not img_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")

    # ===== TRƯỚC tiền xử lý: ảnh gốc =====
    original = Image.open(img_path).convert("RGB")
    orig_arr = np.array(original)  # uint8 [0, 255]

    # ===== SAU tiền xử lý: đúng pipeline preprocess_single_image =====
    import tensorflow as tf
    from src.preprocessing import preprocess_single_image
    processed = preprocess_single_image(
        tf.constant(orig_arr), C.IMG_SIZE).numpy()  # float32 [0, 1]

    # ===== Vẽ hình so sánh =====
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(orig_arr)
    axes[0].set_title(
        f"TRƯỚC tiền xử lý\n{orig_arr.shape[1]}×{orig_arr.shape[0]} px — "
        f"uint8, pixel [{orig_arr.min()}, {orig_arr.max()}]",
        fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(processed)
    axes[1].set_title(
        f"SAU tiền xử lý\n{C.IMG_SIZE}×{C.IMG_SIZE} px — "
        f"float32, pixel [{processed.min():.2f}, {processed.max():.2f}]",
        fontsize=11)
    axes[1].axis("off")

    fig.suptitle("So sánh ảnh trước / sau tiền xử lý "
                 f"(resize {C.IMG_SIZE}×{C.IMG_SIZE} + normalize /255)",
                 fontsize=13)
    fig.tight_layout()

    out_path = C.FIGURES_DIR / f"preprocessing_comparison_{img_path.stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Đã lưu: {out_path}")
    print(f"  Ảnh gốc : {img_path.name}, shape={orig_arr.shape}, dtype={orig_arr.dtype}")
    print(f"  Sau xử lý: shape={processed.shape}, dtype={processed.dtype}")


if __name__ == "__main__":
    main()
