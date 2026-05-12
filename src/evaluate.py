"""Đánh giá model: classification report + confusion matrix.

Cách dùng:
    python -m src.evaluate
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from . import config as C
from .data_loader import (
    gather_filepaths, get_display_names, list_class_folders, make_tf_dataset,
)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.4),
                                    max(6, len(class_names) * 0.35)))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print(">>> Load model và metadata lớp...")
    model = tf.keras.models.load_model(C.MODEL_PATH)
    class_folders = list_class_folders(C.PROCESSED_TRAIN_DIR)
    class_names = get_display_names(class_folders, lang="vi")
    num_classes = len(class_folders)

    print(">>> Load test set từ data/processed/test/...")
    x_te, y_te = gather_filepaths(C.PROCESSED_TEST_DIR, class_folders)
    test_ds = make_tf_dataset(x_te, y_te, num_classes)

    print(">>> Dự đoán...")
    y_true = y_te
    y_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    print(">>> Classification report...")
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=True,
    )
    text_report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4,
    )
    print(text_report)
    with open(C.REPORTS_DIR / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(C.REPORTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(text_report)

    print(">>> Confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, C.FIGURES_DIR / "confusion_matrix.png")
    print(f">>> Lưu kết quả tại: {C.REPORTS_DIR}")


if __name__ == "__main__":
    main()
