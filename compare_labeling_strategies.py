"""
compare_labeling_strategies.py
================================
Bandingkan 4 strategi labeling tanpa training apa-apa:
  - "any"
  - "majority"
  - "threshold" (multiple values: 0.2, 0.3, 0.5)
  - "sliding" (multiple strides)

Output: tabel distribusi label per strategi untuk train, valid, test.
Tujuan: bantu Fikar memilih strategi mana yang menghasilkan distribusi
        paling sehat sebelum jalankan training.

Tidak butuh modifikasi apapun di crop atau features — hanya baca.
"""

import json
from pathlib import Path
from collections import Counter
import argparse


def aggregate_labels(class_per_frame, strategy="majority", threshold=0.3):
    """Aggregasi label per-siswa. Konsisten dengan dataset.py v6."""
    valid_labels = [c for c in class_per_frame if c >= 0]
    if not valid_labels:
        return -1

    n_cheat = sum(1 for c in valid_labels if c == 0)  # YOLO 0 = cheat
    n_total = len(valid_labels)
    ratio   = n_cheat / n_total

    if strategy == "any":
        return 1 if n_cheat > 0 else 0
    elif strategy == "majority":
        return 1 if ratio > 0.5 else 0
    elif strategy == "threshold":
        return 1 if ratio > threshold else 0
    else:
        return -1


def diagnose_per_student(crop_root: str, strategy: str, threshold: float = 0.3):
    """Cek distribusi label per-siswa untuk strategi tertentu."""
    crop_path = Path(crop_root)
    results = {}

    for split in ["train", "valid", "test"]:
        split_dir = crop_path / split
        if not split_dir.exists():
            continue

        labels = []
        for video_dir in sorted(split_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            json_path = video_dir / "labels_per_student.json"
            if not json_path.exists():
                continue
            with open(json_path) as f:
                all_labels = json.load(f)
            for student_id, classes in all_labels.items():
                lbl = aggregate_labels(classes, strategy, threshold)
                if lbl >= 0:
                    labels.append(lbl)

        dist = Counter(labels)
        n_cheat = dist.get(1, 0)
        n_not   = dist.get(0, 0)
        n_total = len(labels)
        ratio_str = f"{n_cheat}:{n_not}"
        balance = f"{n_cheat/n_total*100:.0f}:{n_not/n_total*100:.0f}" if n_total > 0 else "-"
        results[split] = {
            "total": n_total,
            "n_cheat": n_cheat,
            "n_not": n_not,
            "ratio": ratio_str,
            "balance": balance,
        }

    return results


def diagnose_sliding(
    crop_root: str, feature_root: str,
    window_size: int = 8, stride: int = 4,
    strategy: str = "majority", threshold: float = 0.3,
    min_valid_ratio: float = 0.5,
):
    """Cek distribusi label saat pakai sliding window."""
    import numpy as np
    crop_path    = Path(crop_root)
    feature_path = Path(feature_root)
    results = {}

    for split in ["train", "valid", "test"]:
        feature_split = feature_path / split
        crop_split    = crop_path    / split
        if not feature_split.exists():
            continue

        labels = []
        n_students = 0
        for video_dir in sorted(feature_split.iterdir()):
            if not video_dir.is_dir():
                continue
            crop_video_dir = crop_split / video_dir.name
            json_path = crop_video_dir / "labels_per_student.json"
            if not json_path.exists():
                continue
            with open(json_path) as f:
                all_labels = json.load(f)

            for npy_file in sorted(video_dir.glob("*.npy")):
                student_id = npy_file.stem
                student_classes = all_labels.get(student_id, [])
                if not student_classes:
                    continue
                # Untuk efisiensi diagnose: pakai length class_per_frame saja
                # tidak load .npy
                T = len(student_classes)
                n_students += 1

                # Sliding window
                for start in range(0, T - window_size + 1, stride):
                    end = start + window_size
                    window_classes = student_classes[start:end]
                    valid_count = sum(1 for c in window_classes if c >= 0)
                    if valid_count / window_size < min_valid_ratio:
                        continue
                    lbl = aggregate_labels(window_classes, strategy, threshold)
                    if lbl >= 0:
                        labels.append(lbl)

        dist = Counter(labels)
        n_cheat = dist.get(1, 0)
        n_not   = dist.get(0, 0)
        n_total = len(labels)
        balance = f"{n_cheat/n_total*100:.0f}:{n_not/n_total*100:.0f}" if n_total > 0 else "-"
        results[split] = {
            "n_students": n_students,
            "total_windows": n_total,
            "n_cheat": n_cheat,
            "n_not": n_not,
            "balance": balance,
        }

    return results


def print_table(title: str, results: dict, columns: list):
    """Print tabel cantik."""
    print(f"\n  {title}")
    print("  " + "-" * 60)
    header = "  " + " | ".join(f"{c:>12s}" for c in columns)
    print(header)
    print("  " + "-" * 60)
    for split in ["train", "valid", "test"]:
        if split not in results:
            continue
        r = results[split]
        row = "  " + " | ".join(f"{str(r.get(c, '-')):>12s}" for c in columns)
        # Highlight kalau distribusi sangat tidak seimbang
        if r.get("n_not", 0) == 0 or r.get("n_cheat", 0) == 0:
            row += "   ⚠️  SATU KELAS"
        elif r.get("balance"):
            pct = int(r["balance"].split(":")[0])
            if pct < 15 or pct > 85:
                row += "   ⚠️  imbalance >85%"
            elif 30 <= pct <= 70:
                row += "   ✓ balanced"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-root", default="crop")
    parser.add_argument("--feature-root", default="features")
    parser.add_argument("--window-size", type=int, default=8)
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("PERBANDINGAN STRATEGI LABELING")
    print("=" * 65)
    print("\n(YOLO class 0 = cheat, 1 = not_cheat → dibalik di dataset:")
    print(" output label 0 = not_cheat, 1 = cheat)")

    # ─── PER-STUDENT (4 variasi) ─────────────────────────────────
    print("\n" + "─" * 65)
    print("OPSI A: Per-Siswa (1 student = 1 sample)")
    print("─" * 65)

    cols = ["total", "n_cheat", "n_not", "balance"]
    for strat in ["any", "majority"]:
        res = diagnose_per_student(args.crop_root, strat)
        print_table(f"Strategy: '{strat}'", res, cols)

    for thr in [0.2, 0.3, 0.5]:
        res = diagnose_per_student(args.crop_root, "threshold", thr)
        print_table(f"Strategy: 'threshold' (cheat if ratio > {thr})", res, cols)

    # ─── SLIDING WINDOW (3 variasi stride) ───────────────────────
    print("\n" + "─" * 65)
    print(f"OPSI B: Sliding Window (window_size={args.window_size})")
    print("─" * 65)

    cols_sl = ["n_students", "total_windows", "n_cheat", "n_not", "balance"]
    for stride in [args.window_size, args.window_size // 2, args.window_size // 4]:
        if stride < 1:
            continue
        res = diagnose_sliding(
            args.crop_root, args.feature_root,
            window_size=args.window_size, stride=stride,
            strategy="majority",
        )
        print_table(
            f"Sliding (stride={stride}, overlap={(args.window_size-stride)/args.window_size*100:.0f}%, majority)",
            res, cols_sl
        )

    # ─── Rekomendasi ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("REKOMENDASI:")
    print("=" * 65)
    print("  - Pilih strategi yang menghasilkan distribusi 'balanced' di")
    print("    semua split (train, valid, test) — idealnya 30:70 sampai 70:30")
    print("  - Hindari strategi yang ada split '⚠️ SATU KELAS'")
    print("  - Sliding window biasanya menghasilkan dataset lebih besar")
    print("    + distribusi lebih akurat per-window")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
