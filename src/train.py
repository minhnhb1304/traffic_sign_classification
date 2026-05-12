"""Script huấn luyện CNN tự xây cho dataset biển báo VN.

Cách dùng:
    python -m src.train
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from . import config as C
from .data_loader import load_train_val_test, make_tf_dataset
from .model import build_cnn, compile_model
from .preprocessing import build_augmentation_pipeline


def get_callbacks() -> list:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(C.MODEL_PATH), save_best_only=True,
            monitor="val_loss", mode="min", verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min",
            patience=C.EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=C.LR_REDUCE_PATIENCE, min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(C.REPORTS_DIR / "training_log.csv")),
    ]


def compute_class_weights(y_train: np.ndarray, num_classes: int,
                          max_weight: float = 2.0) -> dict:
    """Trọng số cân bằng cho các lớp: weight_i = N / (K * count_i), cap ở max_weight.

    GTSRB cân bằng vừa phải (210–2250 ảnh/lớp, tỉ lệ ~1:11). Cap thấp (2.0) tránh
    over-prediction các lớp ít sample như từng thấy ở lần train đầu (Bicycles
    crossing precision 0.28, Turn left ahead 0.40).
    """
    counts = Counter(int(y) for y in y_train)
    total = sum(counts.values())
    raw = {i: total / (num_classes * counts.get(i, 1)) for i in range(num_classes)}
    return {i: min(w, max_weight) for i, w in raw.items()}


def plot_history(history, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print(">>> Bước 1: Load dataset đã crop (data/processed/)...")
    (x_tr, y_tr), (x_val, y_val), (x_te, y_te), class_folders = load_train_val_test()
    num_classes = len(class_folders)
    print(f"    Số lớp: {num_classes}")
    print(f"    Train: {len(x_tr)} | Val: {len(x_val)} | Test: {len(x_te)}")

    print(">>> Bước 2: Tạo tf.data.Dataset...")
    aug = build_augmentation_pipeline()
    train_ds = make_tf_dataset(x_tr, y_tr, num_classes, shuffle=True, augment_fn=aug)
    val_ds = make_tf_dataset(x_val, y_val, num_classes)
    test_ds = make_tf_dataset(x_te, y_te, num_classes)

    print(">>> Bước 3: Xây và compile model...")
    model = build_cnn(num_classes=num_classes)
    model = compile_model(model)
    model.summary()

    print(">>> Bước 4: Huấn luyện...")
    class_weight = compute_class_weights(y_tr, num_classes)
    print(f"    Class weights: min={min(class_weight.values()):.2f}, "
          f"max={max(class_weight.values()):.2f}")
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=C.EPOCHS, callbacks=get_callbacks(),
        class_weight=class_weight, verbose=1,
    )

    print(">>> Bước 5: Đánh giá trên test set...")
    test_metrics = model.evaluate(test_ds, return_dict=True, verbose=1)
    print(f"    Test metrics: {test_metrics}")

    with open(C.REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    plot_history(history, C.FIGURES_DIR / "training_curves.png")
    print(f">>> Hoàn tất. Model lưu tại: {C.MODEL_PATH}")


if __name__ == "__main__":
    main()
