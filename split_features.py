"""
split_features.py — Memisah Fitur 38-dim ke 3 Kelompok
========================================================
Membuat 3 versi folder features untuk eksperimen perbandingan:

  Kelompok A — Koordinat only (21 dim)
    [0:21]  Raw keypoints (7 kp × 3: x_norm, y_norm, conf)
            5 head kp (nose, eyes, ears) + 2 shoulder

  Kelompok B — Koordinat + Geometric Pose (28 dim)
    [0:21]   Raw keypoints
    [21:24]  Geometric head pose (yaw, pitch, roll)
    [24:26]  Head-body relation (y_relative, size_ratio)
    [26:28]  Visibility flags (n_visible_norm, facing_back_flag)

  Kelompok C — Koordinat + Geometric Pose + Derivatives (38 dim)
    Semua fitur termasuk velocity temporal [28:38]
    (= file .npy original, hanya di-copy)

Output:
  features_A_coord/      ← shape (8, 21)
  features_B_coord_geom/ ← shape (8, 28)
  features_C_full/       ← shape (8, 38)  — copy original
"""

import shutil
import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Definisi indeks fitur (konsisten dengan feature_extractor_v3.py)
# ─────────────────────────────────────────────────────────────────
FEATURE_GROUPS = {
    "A_coord": {
        "indices": list(range(0, 21)),     # [0:21] raw keypoints
        "dim":     21,
        "desc":    "Koordinat only (7 keypoints × 3)",
    },
    "B_coord_geom": {
        "indices": list(range(0, 28)),     # [0:21] + [21:24] + [24:26] + [26:28]
        "dim":     28,
        "desc":    "Koordinat + Geometric Pose + Visibility",
    },
    "C_full": {
        "indices": list(range(0, 38)),     # semua
        "dim":     38,
        "desc":    "Koordinat + Geometric Pose + Derivatives (FULL)",
    },
}


def split_one_npy(npy_path: Path, output_path: Path, indices: list):
    """Baca .npy original, ambil subset kolom, simpan ke output."""
    arr = np.load(str(npy_path))   # (seq_len, 38)
    subset = arr[:, indices]       # (seq_len, len(indices))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), subset.astype(np.float32))
    return subset.shape


def split_features(
    source_root: str,
    target_root_prefix: str = "features",
    groups: list = None,
    copy_labels_json: bool = True,
    crop_root: str = "crop",
):
    """
    Split fitur 38-dim ke 3 folder berbeda.

    Args:
        source_root         : Path ke features/ (output Fase 1)
        target_root_prefix  : Prefix folder output (default "features")
        groups              : List nama grup yang diproses, default semua
        copy_labels_json    : Salin labels_per_student.json juga? (default Yes)
        crop_root           : Path ke crop/ untuk salin labels_per_student.json
    """
    if groups is None:
        groups = list(FEATURE_GROUPS.keys())

    source_path = Path(source_root)
    crop_path   = Path(crop_root)
    if not source_path.exists():
        log.error(f"Folder source tidak ada: {source_path}")
        return

    # Stats untuk verifikasi
    stats = {g: {"files": 0, "shape": None} for g in groups}

    for group_name in groups:
        group_info = FEATURE_GROUPS[group_name]
        indices = group_info["indices"]
        target_path = Path(f"{target_root_prefix}_{group_name}")

        log.info(f"\n{'='*55}")
        log.info(f"GROUP: {group_name}")
        log.info(f"  {group_info['desc']}")
        log.info(f"  Indices: {indices[:5]}...{indices[-3:]} (total {group_info['dim']} dim)")
        log.info(f"  Output:  {target_path}")
        log.info(f"{'='*55}")

        # Hapus target folder lama jika ada (biar tidak campur dengan run sebelumnya)
        if target_path.exists():
            log.info(f"  Folder existing dihapus: {target_path}")
            shutil.rmtree(target_path)
        target_path.mkdir(parents=True, exist_ok=True)

        # Iterasi seluruh struktur features/<split>/<video_id>/*.npy
        for split_dir in sorted(source_path.iterdir()):
            if not split_dir.is_dir():
                continue
            for video_dir in sorted(split_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                for npy_file in sorted(video_dir.glob("*.npy")):
                    rel_path = npy_file.relative_to(source_path)
                    out_file = target_path / rel_path

                    shape = split_one_npy(npy_file, out_file, indices)
                    stats[group_name]["files"] += 1
                    stats[group_name]["shape"] = shape

        # Copy file extraction_stats.json jika ada
        ext_stats = source_path / "extraction_stats.json"
        if ext_stats.exists():
            shutil.copy(ext_stats, target_path / "extraction_stats.json")

        log.info(f"  ✓ {stats[group_name]['files']} file disalin "
                 f"dengan shape {stats[group_name]['shape']}")

    # ⚠️ PENTING: Salin labels_per_student.json
    # Karena dataset.py butuh crop_root untuk baca label,
    # dan kita ingin training pakai label yang SAMA antar grup,
    # maka labels_per_student.json TIDAK perlu duplikasi.
    # Cukup pass --crop-root yang sama ke train.py untuk semua eksperimen.
    if copy_labels_json:
        log.info(f"\n💡 INFO Label:")
        log.info(f"  labels_per_student.json di crop/{crop_path.name}/...")
        log.info(f"  TIDAK diduplikasi (label sama untuk semua grup)")
        log.info(f"  Pakai --crop-root {crop_path} di semua eksperimen train.py")

    # Ringkasan
    log.info(f"\n{'='*55}")
    log.info("RINGKASAN SPLIT FEATURES")
    log.info(f"{'='*55}")
    for g in groups:
        log.info(f"  {g:20s} : {stats[g]['files']:4d} file, shape={stats[g]['shape']}")
    log.info(f"\n  Total folder output: {len(groups)}")
    log.info(f"{'='*55}\n")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split fitur 38-dim ke 3 kelompok untuk eksperimen perbandingan"
    )
    parser.add_argument("--source",  default="features",
                        help="Folder fitur original (38-dim)")
    parser.add_argument("--prefix",  default="features",
                        help="Prefix folder output (default 'features')")
    parser.add_argument("--crop",    default="crop",
                        help="Path ke crop/ untuk baca label")
    parser.add_argument("--groups",  nargs="+",
                        default=["A_coord", "B_coord_geom", "C_full"],
                        help="Pilih grup yang ingin di-generate")

    args = parser.parse_args()

    split_features(
        source_root        = args.source,
        target_root_prefix = args.prefix,
        groups             = args.groups,
        crop_root          = args.crop,
    )
