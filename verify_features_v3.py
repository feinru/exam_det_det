"""
verify_features_v3.py
======================
Verifikasi hasil Fase 1 v3 — multi-variant.

Auto-detect varian dari `variant_meta.json` di feature root.
Jika tidak ada, fallback ke flag --variant atau full (38).

Cek:
- Shape (seq_len, feature_dim) sesuai varian
- Range nilai per kelompok fitur
- Statistik pose detection & facing-back ratio
"""

import json
import numpy as np
from pathlib import Path
import argparse


# Layout per varian (harus konsisten dengan feature_extractor_v3.py)
VARIANT_LAYOUTS = {
    "coord": {
        "dim": 23,
        "raw_kp": list(range(0, 21)),
        "geom": None,
        "body_rel": None,
        "visibility": (21, 23),  # (n_vis_idx, facing_back_idx+1)
        "velocity": None,
    },
    "geom": {
        "dim": 28,
        "raw_kp": list(range(0, 21)),
        "geom": (21, 24),
        "body_rel": (24, 26),
        "visibility": (26, 28),
        "velocity": None,
    },
    "full": {
        "dim": 38,
        "raw_kp": list(range(0, 21)),
        "geom": (21, 24),
        "body_rel": (24, 26),
        "visibility": (26, 28),
        "velocity": (28, 38),
    },
}


def detect_variant(feature_root: Path) -> str:
    """Baca variant_meta.json kalau ada, else default 'full'."""
    meta_path = feature_root / "variant_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        return meta["variant"]
    return "full"


def verify(feature_root: str, expected_seq_len: int = 8,
           variant: str = None):
    feature_path = Path(feature_root)
    errors = []
    stats = []

    if variant is None:
        variant = detect_variant(feature_path)

    layout = VARIANT_LAYOUTS[variant]
    expected_feat_dim = layout["dim"]

    print(f"\n{'='*55}")
    print(f"VERIFIKASI FEATURES — {feature_root}")
    print(f"Varian terdeteksi: {variant.upper()} (dim={expected_feat_dim})")
    print(f"Shape target     : ({expected_seq_len}, {expected_feat_dim})")
    print(f"{'='*55}")

    # Index visibility (n_vis_idx, facing_back_idx) untuk reporting
    n_vis_idx = layout["visibility"][0]
    facing_back_idx = layout["visibility"][1] - 1

    # Index koordinat raw_kp (x dan y)
    raw_kp_x_idx = [i*3 + 0 for i in range(7)]
    raw_kp_y_idx = [i*3 + 1 for i in range(7)]

    total_files, ok_files = 0, 0

    for split_dir in sorted(feature_path.iterdir()):
        if not split_dir.is_dir():
            continue

        for video_dir in sorted(split_dir.iterdir()):
            if not video_dir.is_dir():
                continue

            npy_files = sorted(video_dir.glob("*.npy"))
            if not npy_files:
                continue

            print(f"\n  [{split_dir.name}/{video_dir.name}] — {len(npy_files)} file")

            for npy_path in npy_files:
                total_files += 1
                try:
                    arr = np.load(str(npy_path))

                    if arr.shape != (expected_seq_len, expected_feat_dim):
                        errors.append(f"Shape salah {npy_path.name}: {arr.shape} (expected ({expected_seq_len},{expected_feat_dim}))")
                        continue

                    # Range checks
                    raw_kp_x = arr[:, raw_kp_x_idx]
                    raw_kp_y = arr[:, raw_kp_y_idx]
                    vis_slice = arr[:, layout["visibility"][0]:layout["visibility"][1]]

                    kp_out = ((raw_kp_x < -0.01).any() or (raw_kp_x > 1.01).any() or
                              (raw_kp_y < -0.01).any() or (raw_kp_y > 1.01).any())
                    vis_out = (vis_slice < -0.01).any() or (vis_slice > 1.01).any()

                    issues = []
                    if kp_out: issues.append("kp_range")
                    if vis_out: issues.append("vis_range")

                    if layout["geom"] is not None:
                        pose_slice = arr[:, layout["geom"][0]:layout["geom"][1]]
                        if (pose_slice < -1.05).any() or (pose_slice > 1.05).any():
                            issues.append("pose_range")

                    if layout["velocity"] is not None:
                        vel_slice = arr[:, layout["velocity"][0]:layout["velocity"][1]]
                        if (vel_slice < -0.51).any() or (vel_slice > 0.51).any():
                            issues.append("vel_range")

                    facing_back_ratio = float(arr[:, facing_back_idx].mean())
                    visible_ratio = float(arr[:, n_vis_idx].mean())

                    status = "✓" if not issues else "⚠"
                    label = ""
                    if facing_back_ratio > 0.5:
                        label = "  [mostly facing back]"
                    elif visible_ratio < 0.3:
                        label = "  [low visibility]"

                    print(
                        f"    {status} {npy_path.name:<25s} "
                        f"vis={visible_ratio:.2f} back={facing_back_ratio:.0%}"
                        f"{label}"
                    )

                    if issues:
                        errors.append(f"{npy_path.name}: {issues}")

                    stats.append({
                        "facing_back": facing_back_ratio,
                        "visibility": visible_ratio,
                    })
                    ok_files += 1

                except Exception as e:
                    errors.append(f"Error {npy_path}: {e}")

    print(f"\n{'='*55}")
    print(f"Total: {ok_files}/{total_files} file OK")
    if stats:
        avg_back = np.mean([s["facing_back"] for s in stats])
        avg_vis = np.mean([s["visibility"] for s in stats])
        print(f"Avg facing-back ratio : {avg_back:.1%}")
        print(f"Avg head visibility   : {avg_vis:.1%}")

    if errors:
        print(f"\n⛔ {len(errors)} ERROR (first 10):")
        for e in errors[:10]:
            print(f"   - {e}")
    else:
        print("\n✅ Semua file valid!")

    print(f"{'='*55}\n")
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", default="features_full")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--variant", choices=list(VARIANT_LAYOUTS.keys()), default=None,
                        help="Manual override; default auto-detect dari variant_meta.json")
    parser.add_argument("--all", action="store_true",
                        help="Verifikasi features_coord, features_geom, features_full sekaligus")
    args = parser.parse_args()

    if args.all:
        all_errors = []
        for v in ["coord", "geom", "full"]:
            d = f"features_{v}"
            if Path(d).exists():
                all_errors.extend(verify(d, args.seq_len, v))
            else:
                print(f"\n[skip] {d} tidak ada")
        exit(0 if not all_errors else 1)
    else:
        errors = verify(args.feature_dir, args.seq_len, args.variant)
        exit(0 if not errors else 1)
