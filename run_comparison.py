"""
run_comparison.py — Loop training 3 varian fitur dan kumpulkan hasilnya.

Asumsi: feature sudah diekstrak ke folder features_coord/, features_geom/,
features_full/ via:
    python feature_extractor_v3.py --all-variants

Penggunaan:
    python run_comparison.py
    python run_comparison.py --variants coord geom full --epochs 50
    python run_comparison.py --variants coord full --epochs 30 --skip-missing

Output:
    output/coord/  output/geom/  output/full/
    output/comparison.json  output/comparison.png  output/comparison.md
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def feature_root_exists(variant: str) -> bool:
    return Path(f"features_{variant}").exists() and (
        Path(f"features_{variant}") / "variant_meta.json"
    ).exists()


def run_one(variant: str, epochs: int, batch_size: int, lr: float,
            seed: int, extra_args: list, dry_run: bool):
    """Panggil train.py untuk satu varian."""
    cmd = [
        sys.executable, "train.py",
        "--feature-variant", variant,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--seed", str(seed),
    ] + list(extra_args)

    log.info("\n" + "#" * 60)
    log.info(f"# RUN: variant={variant} (epochs={epochs}, lr={lr})")
    log.info(f"# cmd: {' '.join(cmd)}")
    log.info("#" * 60)

    if dry_run:
        log.info("[dry-run] skip eksekusi")
        return 0

    result = subprocess.run(cmd)
    return result.returncode


def collect_summary(variants: list) -> dict:
    """Kumpulkan summary.json tiap varian."""
    results = {}
    for v in variants:
        summary_path = Path(f"output/{v}/summary.json")
        if not summary_path.exists():
            log.warning(f"summary.json tidak ditemukan untuk {v}: {summary_path}")
            continue
        with open(summary_path) as f:
            results[v] = json.load(f)
    return results


def plot_comparison(results: dict, output_path: str):
    """Bar chart perbandingan metrik antar varian."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        log.warning("matplotlib tidak terinstall, skip plotting comparison.")
        return

    if not results:
        log.warning("Tidak ada hasil untuk di-plot.")
        return

    variants = list(results.keys())
    metrics = ["best_val_acc", "best_val_prec", "best_val_rec", "best_val_f1"]
    metric_labels = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Perbandingan 3 Varian Fitur — Best Validation Metrics",
                 fontsize=14, fontweight="bold")

    colors = {"coord": "#90CAF9", "geom": "#FFB74D", "full": "#81C784"}
    bar_colors = [colors.get(v, "#9E9E9E") for v in variants]

    for ax, metric, label in zip(axes, metrics, metric_labels):
        values = [results[v].get(metric, 0.0) or 0.0 for v in variants]
        bars = ax.bar(variants, values, color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.set_title(label)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.grid(True, axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Grafik perbandingan tersimpan: {output_path}")


def write_markdown_report(results: dict, output_path: str):
    """Tabel markdown ringkasan."""
    if not results:
        return

    lines = [
        "# Perbandingan 3 Varian Fitur",
        "",
        "Hasil training GRU dengan 3 set fitur berbeda.",
        "",
        "| Varian | Dim | n_train | Best Epoch | Val Loss | Val Acc | Val Prec | Val Rec | Val F1 |",
        "|--------|-----|---------|------------|----------|---------|----------|---------|--------|",
    ]
    for variant, s in results.items():
        lines.append(
            f"| {variant} | {s.get('input_dim','?')} | {s.get('n_train','?')} | "
            f"{s.get('best_epoch','?')} | "
            f"{s.get('best_val_loss',0):.4f} | "
            f"{(s.get('best_val_acc') or 0):.4f} | "
            f"{(s.get('best_val_prec') or 0):.4f} | "
            f"{(s.get('best_val_rec') or 0):.4f} | "
            f"{(s.get('best_val_f1') or 0):.4f} |"
        )

    lines += [
        "",
        "Metrik precision/recall/F1 menggunakan **macro-average**.",
        "",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    log.info(f"Laporan markdown tersimpan: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Loop training 3 varian fitur")
    parser.add_argument("--variants", nargs="+",
                        default=["coord", "geom", "full"],
                        choices=["coord", "geom", "full"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip varian yang feature folder-nya belum ada")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-collect", action="store_true",
                        help="Skip training, hanya kumpulkan hasil dari output/")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="Argumen tambahan untuk train.py (setelah --extra)")
    args = parser.parse_args()

    # Filter varian yang feature-nya tersedia
    valid_variants = []
    for v in args.variants:
        if feature_root_exists(v):
            valid_variants.append(v)
        else:
            msg = f"features_{v}/ tidak ada atau belum ada variant_meta.json"
            if args.skip_missing or args.only_collect:
                log.warning(msg + " — di-skip")
            else:
                log.error(msg)
                log.error("Jalankan dulu: python feature_extractor_v3.py --variant " + v)
                sys.exit(1)

    if not valid_variants:
        log.error("Tidak ada varian valid untuk dijalankan.")
        sys.exit(1)

    # Jalankan training
    if not args.only_collect:
        for variant in valid_variants:
            rc = run_one(
                variant, args.epochs, args.batch_size, args.lr,
                args.seed, args.extra, args.dry_run,
            )
            if rc != 0:
                log.error(f"Training gagal untuk varian {variant} (rc={rc}). Lanjut.")

    # Kumpulkan hasil
    results = collect_summary(valid_variants)

    if not results:
        log.warning("Tidak ada hasil untuk diringkas.")
        return

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Ringkasan JSON tersimpan: {out_dir / 'comparison.json'}")

    plot_comparison(results, str(out_dir / "comparison.png"))
    write_markdown_report(results, str(out_dir / "comparison.md"))

    # Print ringkas ke stdout
    print("\n" + "=" * 60)
    print("HASIL PERBANDINGAN")
    print("=" * 60)
    print(f"{'Variant':<8} {'Dim':<5} {'BestEp':<7} {'ValLoss':<9} {'Acc':<7} {'Prec':<7} {'Rec':<7} {'F1':<7}")
    print("-" * 60)
    for v, s in results.items():
        print(f"{v:<8} {s.get('input_dim','?'):<5} {str(s.get('best_epoch','?')):<7} "
              f"{s.get('best_val_loss',0):<9.4f} "
              f"{(s.get('best_val_acc') or 0):<7.4f} "
              f"{(s.get('best_val_prec') or 0):<7.4f} "
              f"{(s.get('best_val_rec') or 0):<7.4f} "
              f"{(s.get('best_val_f1') or 0):<7.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
