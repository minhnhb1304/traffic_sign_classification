"""Tiền xử lý GTSRB (Kaggle format) -> dataset classification đã crop.

Đầu vào (1 trong 2 nguồn):
  - File zip Kaggle (mặc định: GTSRB.zip ở root) — sẽ giải nén vào data/gtsrb_raw/
  - Thư mục đã giải nén chứa Train/, Test/, Train.csv, Test.csv

Đầu ra:
  data/processed/{train,val,test}/<class_folder>/<basename>.png
  models/labels.json            (metadata 43 lớp, theo thứ tự ClassId)
  reports/prepare_data_summary.json

Cách dùng:
    python -m src.prepare_data                          # dùng GTSRB.zip ở root
    python -m src.prepare_data --raw data/gtsrb_raw     # dùng folder đã giải nén
    python -m src.prepare_data --zip path/to/GTSRB.zip
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from . import config as C

VAL_RATIO_FROM_TRAIN = 0.10
PADDING_RATIO = 0.05
NUM_CLASSES = 43


def slugify(text: str) -> str:
    """Chuyển tên lớp sang chuỗi an toàn cho tên thư mục (ASCII, [a-zA-Z0-9_])."""
    nfkd = unicodedata.normalize("NFD", text)
    ascii_only = nfkd.encode("ascii", "ignore").decode()
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_only).strip("_")
    return safe or "unknown"


def read_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_class_metadata() -> list[dict]:
    """Đọc 3 file classes*.txt cho 43 lớp GTSRB, trả về list dict theo idx."""
    codes = read_lines(C.CLASSES_FILE)
    names_en = read_lines(C.CLASSES_EN_FILE)
    names_vi = read_lines(C.CLASSES_VIE_FILE)
    assert len(codes) == len(names_en) == len(names_vi) == NUM_CLASSES, (
        f"Yêu cầu {NUM_CLASSES} dòng/file, thực tế: "
        f"{len(codes)} codes / {len(names_en)} EN / {len(names_vi)} VI"
    )
    classes = []
    for i, (code, en, vi) in enumerate(zip(codes, names_en, names_vi)):
        folder = f"{i:02d}_{slugify(en)}"
        classes.append({"idx": i, "code": code, "name_en": en,
                        "name_vi": vi, "folder": folder})
    return classes


def ensure_extracted(zip_path: Path | None, raw_dir: Path) -> Path:
    """Đảm bảo có thư mục raw chứa Train/, Test/, Train.csv, Test.csv."""
    needed = ["Train", "Test", "Train.csv", "Test.csv"]
    if all((raw_dir / n).exists() for n in needed):
        print(f">>> Đã có dataset giải nén tại: {raw_dir}")
        return raw_dir
    if zip_path is None or not zip_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {raw_dir} hoặc zip {zip_path}. "
            "Hãy tải GTSRB.zip về root hoặc truyền --zip / --raw."
        )
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f">>> Giải nén {zip_path} -> {raw_dir} (~600MB, mất vài phút)...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw_dir)
    return raw_dir


def parse_csv(csv_path: Path) -> list[dict]:
    """Đọc Train.csv hoặc Test.csv. Trả về list dict có ClassId (int), Path, Roi.*"""
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "class_id": int(r["ClassId"]),
                "path": r["Path"],
                "x1": int(r["Roi.X1"]), "y1": int(r["Roi.Y1"]),
                "x2": int(r["Roi.X2"]), "y2": int(r["Roi.Y2"]),
            })
    return rows


def stratified_train_val_split(rows: list[dict], val_ratio: float, seed: int):
    """Tách val từ train theo từng lớp (stratified)."""
    by_class: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["class_id"]].append(r)
    rng = random.Random(seed)
    train, val = [], []
    for cls, items in by_class.items():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def crop_with_roi(img: Image.Image, x1: int, y1: int, x2: int, y2: int) -> Image.Image:
    """Crop bbox tuyệt đối kèm padding 5%."""
    W, H = img.size
    pw = int((x2 - x1) * PADDING_RATIO)
    ph = int((y2 - y1) * PADDING_RATIO)
    return img.crop((max(0, x1 - pw), max(0, y1 - ph),
                     min(W, x2 + pw), min(H, y2 + ph)))


def process_split(rows: list[dict], split_name: str, raw_dir: Path,
                  classes: list[dict]) -> dict:
    out_root = C.PROCESSED_DIR / split_name
    counts = {c["folder"]: 0 for c in classes}
    skipped = 0
    for r in tqdm(rows, desc=f"[{split_name}]"):
        src = raw_dir / r["path"]
        if not src.exists():
            skipped += 1
            continue
        try:
            img = Image.open(src).convert("RGB")
        except Exception:
            skipped += 1
            continue
        crop = crop_with_roi(img, r["x1"], r["y1"], r["x2"], r["y2"])
        cls = classes[r["class_id"]]
        cls_dir = out_root / cls["folder"]
        cls_dir.mkdir(parents=True, exist_ok=True)
        crop.save(cls_dir / Path(r["path"]).name, format="PNG")
        counts[cls["folder"]] += 1
    return {"counts": counts, "skipped": skipped, "total": sum(counts.values())}


def clean_processed_dir() -> None:
    """Xóa data/processed/{train,val,test} nếu đã tồn tại để tránh trộn dữ liệu cũ."""
    for split in ("train", "val", "test"):
        d = C.PROCESSED_DIR / split
        if d.exists():
            shutil.rmtree(d)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=C.GTSRB_ZIP_PATH,
                        help="Đường dẫn GTSRB.zip (mặc định: <root>/GTSRB.zip)")
    parser.add_argument("--raw", type=Path, default=C.GTSRB_RAW_DIR,
                        help="Đường dẫn folder đã giải nén (mặc định: data/gtsrb_raw/)")
    args = parser.parse_args()

    classes = load_class_metadata()
    print(f">>> Số lớp: {len(classes)}")

    raw_dir = ensure_extracted(args.zip, args.raw)

    print(">>> Đọc Train.csv & Test.csv...")
    train_rows = parse_csv(raw_dir / "Train.csv")
    test_rows = parse_csv(raw_dir / "Test.csv")
    print(f"    Train+Val: {len(train_rows)} ảnh | Test: {len(test_rows)} ảnh")

    train_split, val_split = stratified_train_val_split(
        train_rows, VAL_RATIO_FROM_TRAIN, C.SEED)
    print(f"    Sau stratified split: train={len(train_split)} val={len(val_split)}")

    print(">>> Dọn data/processed/ cũ...")
    clean_processed_dir()

    summary = {}
    for name, rows in [("train", train_split), ("val", val_split), ("test", test_rows)]:
        summary[name] = process_split(rows, name, raw_dir, classes)

    with open(C.LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    print(f">>> Đã lưu metadata: {C.LABELS_PATH}")

    print("\n=== Tổng kết ===")
    for split, info in summary.items():
        print(f"  {split:5s}: {info['total']} ảnh (skipped: {info['skipped']})")

    out_summary = C.REPORTS_DIR / "prepare_data_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f">>> Tóm tắt chi tiết: {out_summary}")


if __name__ == "__main__":
    main()
