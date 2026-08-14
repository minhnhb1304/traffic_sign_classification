"""Chuyển model Keras (custom_cnn_v1.keras) sang TensorFlow Lite FP16.

Model được huấn luyện với input đã chuẩn hóa [0,1] (img/255.0) và KHÔNG có
lớp Rescaling bên trong. Phía Android (TFLiteHelper) cũng chuẩn hóa [0,1]
bằng NormalizeOp(0, 255) trước khi suy luận, nên file .tflite giữ nguyên
input [0,1] — không thêm bước tiền xử lý vào trong model.

Lượng tử hóa FP16: chỉ ép trọng số (weights) về float16 → giảm ~50% dung
lượng model, tăng tốc trên thiết bị di động, độ chính xác gần như giữ nguyên.
Input/output vẫn là float32 nên không cần đổi code Android.

Cách chạy:
    python scripts/convert_to_tflite.py
    python scripts/convert_to_tflite.py --no-fp16        # bản float32 thường
    python scripts/convert_to_tflite.py --model <path> --out <path>
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

# Đường dẫn mặc định theo cấu trúc project
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DEFAULT_MODEL = MODELS_DIR / "custom_cnn_v1.keras"
DEFAULT_OUT = MODELS_DIR / "custom_cnn_v1_fp16.tflite"


def convert(model_path: Path, out_path: Path, use_fp16: bool = True) -> None:
    print(f"[1/4] Nap model Keras: {model_path}")
    model = tf.keras.models.load_model(model_path)

    in_shape = model.input_shape
    out_shape = model.output_shape
    print(f"      Input shape : {in_shape}")
    print(f"      Output shape: {out_shape}")

    print("[2/4] Khoi tao TFLiteConverter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if use_fp16:
        # Luong tu hoa FP16: ep trong so ve float16
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("      Che do: FP16 (weights float16, I/O float32)")
    else:
        print("      Che do: float32 (khong luong tu hoa)")

    print("[3/4] Dang convert...")
    tflite_model = converter.convert()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)

    keras_mb = model_path.stat().st_size / (1024 * 1024)
    tflite_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[4/4] Da luu: {out_path}")
    print(f"      Keras  : {keras_mb:6.2f} MB")
    print(f"      TFLite : {tflite_mb:6.2f} MB")

    verify(model, tflite_model, in_shape)


def verify(keras_model, tflite_bytes: bytes, in_shape) -> None:
    """So sanh ket qua Keras vs TFLite tren 1 anh ngau nhien [0,1]."""
    print("\n[verify] So sanh Keras vs TFLite tren input ngau nhien...")
    h, w, c = in_shape[1], in_shape[2], in_shape[3]
    sample = np.random.rand(1, h, w, c).astype(np.float32)  # gia tri [0,1]

    keras_out = keras_model.predict(sample, verbose=0)[0]

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    interpreter.set_tensor(in_det["index"], sample)
    interpreter.invoke()
    tflite_out = interpreter.get_tensor(out_det["index"])[0]

    k_idx, t_idx = int(np.argmax(keras_out)), int(np.argmax(tflite_out))
    max_diff = float(np.max(np.abs(keras_out - tflite_out)))
    print(f"      Keras  argmax = {k_idx} (p={keras_out[k_idx]:.4f})")
    print(f"      TFLite argmax = {t_idx} (p={tflite_out[t_idx]:.4f})")
    print(f"      Chenh lech xac suat lon nhat = {max_diff:.6f}")
    print("      => " + ("OK, ket qua khop." if k_idx == t_idx
                          else "CANH BAO: argmax khac nhau!"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Keras -> TFLite FP16")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Duong dan file .keras")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Duong dan file .tflite xuat ra")
    parser.add_argument("--no-fp16", action="store_true",
                        help="Tat luong tu hoa FP16 (xuat float32)")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Khong tim thay model: {args.model}")

    convert(args.model, args.out, use_fp16=not args.no_fp16)


if __name__ == "__main__":
    main()
