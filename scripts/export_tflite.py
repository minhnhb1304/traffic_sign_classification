"""Xuất Keras model sang TFLite cho app Android.

Sinh ra (mặc định) trong models/tflite/:
    custom_cnn_v1_fp32.tflite
    custom_cnn_v1_fp16.tflite
    custom_cnn_v1_int8.tflite        (chỉ khi có data/processed/val/)
    labels_android.json              (rút gọn từ models/labels.json)

Cách dùng:
    python -m scripts.export_tflite                     # xuất tất cả biến thể có thể
    python -m scripts.export_tflite --variants fp16     # chỉ FP16
    python -m scripts.export_tflite --rep-samples 300   # số ảnh cho INT8 calibration
"""
from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
from PIL import Image

from src import config as C

OUT_DIR = C.MODELS_DIR / "tflite"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_VARIANTS = ("fp32", "fp16", "int8")


def _read_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((C.IMG_SIZE, C.IMG_SIZE))
    return (np.asarray(img, dtype=np.float32) / 255.0)


def _collect_sample_paths(split_dir: Path, max_samples: int) -> list[Path]:
    if not split_dir.exists():
        return []
    paths: list[Path] = []
    for cdir in sorted(split_dir.iterdir()):
        if cdir.is_dir():
            paths.extend(sorted(cdir.glob("*.png"))[:5])
            paths.extend(sorted(cdir.glob("*.jpg"))[:5])
    np.random.default_rng(C.SEED).shuffle(paths)
    return paths[:max_samples]


def _representative_dataset(paths: list[Path]):
    def gen():
        for p in paths:
            arr = _read_image(p)[None, ...]
            yield [arr.astype(np.float32)]
    return gen


def convert(model: tf.keras.Model, variant: str, rep_paths: list[Path]) -> Path | None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    out_path = OUT_DIR / f"{C.MODEL_NAME}_{variant}.tflite"

    if variant == "fp32":
        pass
    elif variant == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif variant == "int8":
        if not rep_paths:
            print(f"[skip] INT8 cần representative dataset (data/processed/val/) — bỏ qua.")
            return None
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _representative_dataset(rep_paths)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    else:
        raise ValueError(f"Unknown variant: {variant}")

    tflite_bytes = converter.convert()
    out_path.write_bytes(tflite_bytes)
    print(f"[ok]   {variant.upper():>4}  →  {out_path.name}  ({out_path.stat().st_size/1024:.1f} KB)")
    return out_path


def verify(tflite_path: Path, keras_model: tf.keras.Model, sample_paths: list[Path]) -> dict:
    if not sample_paths:
        return {"path": tflite_path.name, "skipped": True}

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    n = min(len(sample_paths), 100)
    arrs = np.stack([_read_image(p) for p in sample_paths[:n]], axis=0)
    keras_pred = keras_model.predict(arrs, batch_size=32, verbose=0).argmax(axis=1)

    tfl_pred = np.empty(n, dtype=np.int64)
    max_diff = 0.0
    for i in range(n):
        x = arrs[i:i+1]
        if in_det["dtype"] == np.uint8:
            scale, zp = in_det["quantization"]
            x_q = np.clip(x / scale + zp, 0, 255).astype(np.uint8)
            interp.set_tensor(in_det["index"], x_q)
        else:
            interp.set_tensor(in_det["index"], x.astype(in_det["dtype"]))
        interp.invoke()
        out = interp.get_tensor(out_det["index"])[0]
        if out_det["dtype"] == np.uint8:
            scale, zp = out_det["quantization"]
            out = (out.astype(np.float32) - zp) * scale
        tfl_pred[i] = int(np.argmax(out))
        max_diff = max(max_diff, float(np.max(np.abs(out - keras_model.predict(x, verbose=0)[0]))))

    match = float(np.mean(tfl_pred == keras_pred)) * 100.0
    return {"path": tflite_path.name,
            "size_kb": round(tflite_path.stat().st_size / 1024, 1),
            "samples": n, "match_with_keras_pct": round(match, 2),
            "max_abs_prob_diff": round(max_diff, 5)}


def export_labels_android() -> Path:
    with open(C.LABELS_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    compact = [{"idx": m["idx"], "name_vi": m["name_vi"], "name_en": m["name_en"]} for m in meta]
    out = OUT_DIR / "labels_android.json"
    out.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok]   labels →  {out.name}  ({len(compact)} classes)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(ALL_VARIANTS), choices=ALL_VARIANTS)
    ap.add_argument("--rep-samples", type=int, default=200,
                    help="Số ảnh cho INT8 representative dataset (lấy từ data/processed/val/).")
    ap.add_argument("--verify-samples", type=int, default=200,
                    help="Số ảnh dùng để so prediction TFLite vs Keras (lấy từ data/processed/test/).")
    args = ap.parse_args()

    print(f">>> Loading Keras model: {C.MODEL_PATH}")
    model = tf.keras.models.load_model(C.MODEL_PATH)
    print(f"    Output shape: {model.output_shape}, params: {model.count_params():,}")

    rep_paths = _collect_sample_paths(C.PROCESSED_VAL_DIR, args.rep_samples)
    test_paths = _collect_sample_paths(C.PROCESSED_TEST_DIR, args.verify_samples)
    print(f"    Representative samples (val): {len(rep_paths)}")
    print(f"    Verify samples (test):        {len(test_paths)}")

    print("\n>>> Converting...")
    produced: list[Path] = []
    for v in args.variants:
        p = convert(model, v, rep_paths)
        if p is not None:
            produced.append(p)

    print("\n>>> Verifying...")
    report = [verify(p, model, test_paths) for p in produced]
    for r in report:
        print(f"    {r}")

    export_labels_android()
    (OUT_DIR / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n>>> Done. Artifacts in: {OUT_DIR}")


if __name__ == "__main__":
    main()
