"""Lọc dataset_VN từ ~3.216 ảnh xuống ~2.000 ảnh để phục vụ báo cáo.

Mục tiêu lọc:
  1) Khử trùng lặp frame liên tiếp (stride ID).
  2) Bảo toàn 100% ảnh chứa lớp hiếm (< RARE_THRESHOLD box).
  3) Cap các lớp đông qua quota mềm để cân bằng phân bố.
  4) Tổng số ảnh cuối ≈ TARGET_TOTAL.
  5) Chia lại train/test 80/20 (seed cố định).

Cách dùng:
    python -m scripts.build_filtered_dataset
    python -m scripts.build_filtered_dataset --target 2000 --quota 250
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data" / "dataset_VN"
DST_DIR = ROOT / "data" / "dataset_VN_filtered"

SEED = 42
TARGET_TOTAL = 2000
STRIDE = 3
RARE_THRESHOLD = 50
QUOTA_PER_COMMON_CLASS = 250
TRAIN_RATIO = 0.8


def read_label_classes(p: Path) -> list[int]:
    """Đọc file YOLO label, trả về list class_id (rỗng nếu file rỗng/không tồn tại)."""
    if not p.exists():
        return []
    out: list[int] = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(int(ln.split()[0]))
        except (ValueError, IndexError):
            continue
    return out


def box_counts(stems: set[str], img_cls: dict[str, list[int]]) -> Counter:
    c: Counter = Counter()
    for s in stems:
        c.update(img_cls[s])
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET_TOTAL)
    ap.add_argument("--quota", type=int, default=QUOTA_PER_COMMON_CLASS)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--rare", type=int, default=RARE_THRESHOLD)
    args = ap.parse_args()

    rng = random.Random(SEED)
    src_img = SRC_DIR / "images"
    src_lbl = SRC_DIR / "labels"

    print(">>> Bước 1: Đọc nhãn nguồn...")
    all_imgs = sorted(src_img.glob("*.jpg"))
    img_cls: dict[str, list[int]] = {p.stem: read_label_classes(src_lbl / f"{p.stem}.txt")
                                     for p in all_imgs}
    orig_counts = box_counts(set(img_cls), img_cls)
    rare = {c for c, n in orig_counts.items() if n < args.rare}
    print(f"    Ảnh: {len(all_imgs)} | tổng box: {sum(orig_counts.values())} "
          f"| lớp hiếm (<{args.rare}): {sorted(rare)}")

    print(f">>> Bước 2: Stride={args.stride} khử frame liên tiếp...")
    stride_kept = {all_imgs[i].stem for i in range(0, len(all_imgs), args.stride)}
    print(f"    Sau stride: {len(stride_kept)} ảnh")

    print(">>> Bước 3: Bảo toàn ảnh chứa lớp hiếm...")
    rare_imgs = {s for s, cs in img_cls.items() if any(c in rare for c in cs)}
    selected = stride_kept | rare_imgs
    print(f"    Thêm {len(rare_imgs - stride_kept)} ảnh hiếm → {len(selected)} ảnh")

    print(f">>> Bước 4: Cap lớp đông tại quota {args.quota} box/lớp...")
    cur = box_counts(selected, img_cls)
    pool = list(selected)
    rng.shuffle(pool)
    for s in pool:
        if len(selected) <= args.target:
            break
        cs = img_cls[s]
        if not cs or any(c in rare for c in cs):
            continue
        if all(cur[c] > args.quota for c in cs):
            selected.discard(s)
            for c in cs:
                cur[c] -= 1
    print(f"    Sau cap: {len(selected)} ảnh")

    print(f">>> Bước 5: Điều chỉnh tổng số = {args.target}...")
    if len(selected) > args.target:
        droppable = [s for s in selected
                     if img_cls[s] and not any(c in rare for c in img_cls[s])]
        rng.shuffle(droppable)
        for s in droppable:
            if len(selected) <= args.target:
                break
            selected.discard(s)
    elif len(selected) < args.target:
        remaining = [s for s in img_cls if s not in selected]
        rng.shuffle(remaining)
        for s in remaining:
            if len(selected) >= args.target:
                break
            selected.add(s)
    final = sorted(selected, key=lambda s: int(s))
    print(f"    Cuối: {len(final)} ảnh")

    print(">>> Bước 6: Copy ảnh + nhãn sang folder mới...")
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    (DST_DIR / "images").mkdir(parents=True)
    (DST_DIR / "labels").mkdir(parents=True)
    for stem in final:
        shutil.copy2(src_img / f"{stem}.jpg", DST_DIR / "images" / f"{stem}.jpg")
        lbl = src_lbl / f"{stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, DST_DIR / "labels" / f"{stem}.txt")
    for fname in ("classes.txt", "classes_en.txt", "classes_vie.txt"):
        if (SRC_DIR / fname).exists():
            shutil.copy2(SRC_DIR / fname, DST_DIR / fname)

    print(">>> Bước 7: Chia lại train/test 80/20...")
    sp = DST_DIR / "split_dataset"
    sp.mkdir(parents=True)
    shuf = list(final)
    rng.shuffle(shuf)
    n_tr = int(len(shuf) * TRAIN_RATIO)
    train = sorted(shuf[:n_tr], key=lambda s: int(s))
    test = sorted(shuf[n_tr:], key=lambda s: int(s))
    (sp / "train_files.txt").write_text("\n".join(f"{s}.jpg" for s in train) + "\n",
                                        encoding="utf-8")
    (sp / "test_files.txt").write_text("\n".join(f"{s}.jpg" for s in test) + "\n",
                                       encoding="utf-8")

    final_counts = box_counts(set(final), img_cls)
    empty = sum(1 for s in final if not img_cls[s])
    print(f"\n=== Tổng kết ===")
    print(f"  Output:       {DST_DIR}")
    print(f"  Ảnh:          {len(final)}  (negative: {empty})")
    print(f"  Tổng box:     {sum(final_counts.values())}")
    print(f"  Train / Test: {len(train)} / {len(test)}")
    print(f"\n  Phân bố lớp (trước → sau):")
    print(f"  {'class':>5} | {'before':>6} | {'after':>5}")
    print("  " + "-" * 24)
    for c in sorted(orig_counts):
        print(f"  {c:>5} | {orig_counts[c]:>6} | {final_counts.get(c, 0):>5}")


if __name__ == "__main__":
    main()
