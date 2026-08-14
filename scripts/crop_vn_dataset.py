"""Crop bounding box từ dataset VN (YOLO format) -> dataset classification.

Đầu vào:
  data/dataset_VN_filtered/
    ├── images/<stem>.jpg
    ├── labels/<stem>.txt              (YOLO: <cls> <cx> <cy> <w> <h>, normalized)
    ├── classes.txt, classes_en.txt, classes_vie.txt
    └── split_dataset/{train,test}_files.txt

Đầu ra:
  data/processed_vn/
    ├── train/<idx:02d>_<slug_en>/<stem>_b<idx>.png
    ├── test/ <idx:02d>_<slug_en>/<stem>_b<idx>.png
    ├── labels_vn.json                 (metadata 52 lớp)
    └── summary.json                   (số ảnh mỗi lớp/split)

Padding 5% nhất quán với src/prepare_data.py (GTSRB). Script độc lập, KHÔNG
sửa pipeline chính.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "data" / "dataset_VN_filtered"
OUT_DIR = ROOT_DIR / "data" / "processed_vn"
PADDING_RATIO = 0.05


def slugify(text: str) -> str:
    nfkd = unicodedata.normalize("NFD", text)
    ascii_only = nfkd.encode("ascii", "ignore").decode()
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_only).strip("_")
    return safe or "unknown"


def read_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_classes(src_dir: Path) -> list[dict]:
    codes = read_lines(src_dir / "classes.txt")
    names_en = read_lines(src_dir / "classes_en.txt")
    names_vi = read_lines(src_dir / "classes_vie.txt")
    assert len(codes) == len(names_en) == len(names_vi), (
        f"Lệch số dòng: {len(codes)}/{len(names_en)}/{len(names_vi)}"
    )
    classes = []
    for i, (code, en, vi) in enumerate(zip(codes, names_en, names_vi)):
        folder = f"{i:02d}_{slugify(en)}"
        classes.append({"idx": i, "code": code, "name_en": en,
                        "name_vi": vi, "folder": folder})
    return classes


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float,
                 W: int, H: int) -> tuple[int, int, int, int]:
    x1 = int(round((cx - w / 2) * W))
    y1 = int(round((cy - h / 2) * H))
    x2 = int(round((cx + w / 2) * W))
    y2 = int(round((cy + h / 2) * H))
    return x1, y1, x2, y2


def crop_with_padding(img: Image.Image, x1: int, y1: int, x2: int, y2: int) -> Image.Image:
    W, H = img.size
    pw = int((x2 - x1) * PADDING_RATIO)
    ph = int((y2 - y1) * PADDING_RATIO)
    return img.crop((max(0, x1 - pw), max(0, y1 - ph),
                     min(W, x2 + pw), min(H, y2 + ph)))


def process_split(stems: list[str], split_name: str,
                  src_dir: Path, out_dir: Path,
                  classes: list[dict], min_size: int) -> dict:
    counts: dict[str, int] = defaultdict(int)
    n_boxes = 0
    n_skipped_small = 0
    n_skipped_missing = 0
    for stem in tqdm(stems, desc=f"[{split_name}]"):
        img_path = src_dir / "images" / f"{stem}.jpg"
        lbl_path = src_dir / "labels" / f"{stem}.txt"
        if not img_path.exists() or not lbl_path.exists():
            n_skipped_missing += 1
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            n_skipped_missing += 1
            continue
        W, H = img.size
        lines = read_lines(lbl_path)
        for box_idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, W, H)
            if (x2 - x1) < min_size or (y2 - y1) < min_size:
                n_skipped_small += 1
                continue
            crop = crop_with_padding(img, x1, y1, x2, y2)
            cls_dir = out_dir / split_name / classes[cls]["folder"]
            cls_dir.mkdir(parents=True, exist_ok=True)
            crop.save(cls_dir / f"{stem}_b{box_idx}.png", format="PNG")
            counts[classes[cls]["folder"]] += 1
            n_boxes += 1
    return {"total": n_boxes, "skipped_small": n_skipped_small,
            "skipped_missing": n_skipped_missing, "counts": dict(counts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=SRC_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--min-size", type=int, default=8,
                        help="Bỏ qua bbox nhỏ hơn min-size pixel (cạnh)")
    args = parser.parse_args()

    classes = load_classes(args.src)
    print(f">>> Số lớp: {len(classes)}")

    train_stems = [Path(p).stem for p in read_lines(args.src / "split_dataset" / "train_files.txt")]
    test_stems = [Path(p).stem for p in read_lines(args.src / "split_dataset" / "test_files.txt")]
    print(f">>> Train ảnh: {len(train_stems)} | Test ảnh: {len(test_stems)}")

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "train": process_split(train_stems, "train", args.src, args.out, classes, args.min_size),
        "test": process_split(test_stems, "test", args.src, args.out, classes, args.min_size),
    }

    (args.out / "labels_vn.json").write_text(
        json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Tổng kết ===")
    for split, info in summary.items():
        print(f"  {split:5s}: {info['total']} crop  "
              f"(skipped_small={info['skipped_small']}, missing={info['skipped_missing']})")
    print(f">>> Output: {args.out}")


if __name__ == "__main__":
    main()
