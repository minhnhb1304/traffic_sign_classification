"""Trích N ảnh/class từ GTSRB Test set ra `data/test_samples/` để demo nhanh.

Đọc trực tiếp từ GTSRB.zip — KHÔNG cần giải nén toàn bộ 612 MB ra disk.
Crop theo bbox của Test.csv với padding 5% (giống `src/prepare_data.py`).

Cách dùng:
    python -m scripts.make_test_samples                       # 5 ảnh/class
    python -m scripts.make_test_samples --per-class 3
    python -m scripts.make_test_samples --zip path/to/GTSRB.zip --out data/demo/
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image
from tqdm import tqdm

from src import config as C
from src.prepare_data import (
    PADDING_RATIO,
    crop_with_roi,
    load_class_metadata,
)


def find_csv_member(zf: zipfile.ZipFile, basename: str) -> str:
    """Tìm path tới Test.csv (hoặc Train.csv) trong zip — chấp nhận có prefix folder."""
    candidates = [n for n in zf.namelist()
                  if Path(n).name.lower() == basename.lower()]
    if not candidates:
        raise FileNotFoundError(f"Không tìm thấy {basename} trong zip.")
    candidates.sort(key=len)
    return candidates[0]


def parse_csv_from_zip(zf: zipfile.ZipFile, csv_name: str) -> list[dict]:
    raw = zf.read(csv_name).decode("utf-8")
    reader = csv.DictReader(io.StringIO(raw))
    rows = []
    for r in reader:
        rows.append({
            "class_id": int(r["ClassId"]),
            "path": r["Path"],
            "x1": int(r["Roi.X1"]), "y1": int(r["Roi.Y1"]),
            "x2": int(r["Roi.X2"]), "y2": int(r["Roi.Y2"]),
        })
    return rows


def resolve_image_in_zip(zf: zipfile.ZipFile, csv_dir: str, csv_path: str) -> str | None:
    """Map cột Path trong CSV → tên file thực trong zip (zip có thể có prefix folder)."""
    rel = csv_path.replace("\\", "/")
    candidates = [
        rel,
        f"{csv_dir}/{rel}" if csv_dir else rel,
    ]
    names = set(zf.namelist())
    for c in candidates:
        if c in names:
            return c
    suffix = "/" + rel
    for n in names:
        if n.endswith(suffix):
            return n
    return None


def pick_samples(rows: list[dict], per_class: int, seed: int) -> list[dict]:
    by_cls: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_cls[r["class_id"]].append(r)
    rng = random.Random(seed)
    picked = []
    for cls in sorted(by_cls):
        items = by_cls[cls][:]
        rng.shuffle(items)
        picked.extend(items[:per_class])
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=C.GTSRB_ZIP_PATH,
                    help=f"Đường dẫn GTSRB.zip (mặc định: {C.GTSRB_ZIP_PATH})")
    ap.add_argument("--per-class", type=int, default=5,
                    help="Số ảnh lấy mỗi lớp (mặc định 5)")
    ap.add_argument("--out", type=Path, default=C.DATA_DIR / "test_samples",
                    help="Thư mục output (mặc định: data/test_samples/)")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--clean", action="store_true",
                    help="Xoá output cũ trước khi tạo mới")
    args = ap.parse_args()

    if not args.zip.exists():
        raise FileNotFoundError(f"Không tìm thấy {args.zip}")

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    classes = load_class_metadata()
    print(f">>> Số lớp: {len(classes)}")

    print(f">>> Đọc Test.csv từ: {args.zip}")
    with zipfile.ZipFile(args.zip) as zf:
        csv_name = find_csv_member(zf, "Test.csv")
        csv_dir = str(Path(csv_name).parent).replace("\\", "/")
        rows = parse_csv_from_zip(zf, csv_name)
        print(f"    Test.csv: {len(rows)} ảnh, prefix='{csv_dir}'")

        picked = pick_samples(rows, args.per_class, args.seed)
        print(f">>> Sẽ trích {len(picked)} ảnh "
              f"({args.per_class}/class × {len(classes)} class)")

        counts: dict[str, int] = defaultdict(int)
        skipped = 0
        for r in tqdm(picked, desc="extract"):
            zip_member = resolve_image_in_zip(zf, csv_dir, r["path"])
            if zip_member is None:
                skipped += 1
                continue
            try:
                with zf.open(zip_member) as fh:
                    img = Image.open(fh).convert("RGB")
                    img.load()
            except Exception:
                skipped += 1
                continue
            crop = crop_with_roi(img, r["x1"], r["y1"], r["x2"], r["y2"])
            cls = classes[r["class_id"]]
            cls_dir = args.out / cls["folder"]
            cls_dir.mkdir(parents=True, exist_ok=True)
            crop.save(cls_dir / Path(r["path"]).name, format="PNG")
            counts[cls["folder"]] += 1

    total = sum(counts.values())
    total_kb = sum(p.stat().st_size for p in args.out.rglob("*.png")) / 1024
    print(f"\n=== Tổng kết ===")
    print(f"  Output: {args.out}")
    print(f"  Tổng ảnh: {total}  (skipped: {skipped})")
    print(f"  Dung lượng: {total_kb:.1f} KB ({total_kb/1024:.2f} MB)")
    print(f"  Padding bbox: {PADDING_RATIO*100:.0f}% (giống prepare_data.py)")
    empty = [c["folder"] for c in classes if counts[c["folder"]] == 0]
    if empty:
        print(f"  ⚠ Lớp KHÔNG có ảnh: {empty}")


if __name__ == "__main__":
    main()
