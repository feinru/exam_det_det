"""
compare_features.py — Eksperimen Perbandingan 3 Kelompok Fitur
================================================================
Jalankan 3 training berurutan untuk:
  A) Koordinat only        (21-dim)
  B) Koor + Geometric Pose (28-dim)
  C) Full (Koor+Geom+Vel)  (38-dim)

Semua pakai SLIDING WINDOW + MAJORITY labeling (rekomendasi).
"""

import json
import logging
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


EXPERIMENTS = [
    {
        "name":         "A_coord",
        "feature_root": "features_A_coord",
        "feature_dim":  21,
        "output_dir":   "output_A_coord",
        "label":        "A: Koordinat (21d)",
        "color":        "#1976D2",
    },
    {
        "name":         "B_coord_geom",
        "feature_root": "features_B_coord_geom",
        "feature_dim":  28,
        "output_dir":   "output_B_coord_geom",
        "label":        "B: Koor+Geometric (28d)",
        "color":        "#388E3C",
    },
    {
        "name":         "C_full",
        "feature_root": "features_C_full",
        "feature_dim":  38,
        "output_dir":   "output_C_full",
        "label":        "C: Full (38d)",
        "color":        "#D32F2F",
    },
]


def run_experiment(exp: dict, args):
    log.info(f"\n{'='*60}")
    log.info(f"EKSPERIMEN: {exp['name']}  ({exp['label']})")
    log.info(f"{'='*60}")

    cmd = [
        sys.executable, "train.py",
        "--feature-root", exp["feature_root"],
        "--crop-root",    args.crop_root,
        "--output-dir",   exp["output_dir"],
        "--epochs",       str(args.epochs),
        "--batch-size",   str(args.batch_size),
        "--seq-len",      str(args.seq_len),
        "--stride",       str(args.stride),
        "--hidden-dim",   str(args.hidden_dim),
        "--lr",           str(args.lr),
        "--seed",         str(args.seed),
        "--labeling",     args.labeling,
        "--patience",     str(args.patience),
    ]

    # use_sliding default ON di train.py, tidak perlu flag
    # Tapi kalau user request --no-sliding lewat compare_features, teruskan
    if args.no_sliding:
        cmd.append("--no-sliding")
    if args.weighted_sampler:
        cmd.append("--weighted-sampler")
    if args.no_pos_weight:
        cmd.append("--no-pos-weight")

    log.info(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error(f"Training {exp['name']} gagal (exit code {result.returncode})")
        return False
    return True


def load_history(output_dir: str):
    history_path = Path(output_dir) / "training_history.json"
    if not history_path.exists():
        log.warning(f"History tidak ditemukan: {history_path}")
        return None
    with open(history_path) as f:
        return json.load(f)


def plot_comparison(experiments_with_history: list, output_path: str):
    fig = plt.figure(figsize=(16, 20))
    fig.suptitle(
        "Perbandingan Feature Groups — Cheating Detection GRU\n"
        "(Train vs Val | Macro Metrics + Recall@Cheat)",
        fontsize=15, fontweight="bold"
    )
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    panels = [
        ("val_loss",       "Loss (Val)",                None),
        ("val_acc",        "Accuracy (Val)",            (0, 1.05)),
        ("val_prec",       "Precision (Val, macro)",    (0, 1.05)),
        ("val_rec",        "Recall (Val, macro)",       (0, 1.05)),
        ("val_f1",         "F1 (Val, macro)",           (0, 1.05)),
        ("val_rec_cheat",  "Recall@Cheat (Val) — KEY",  (0, 1.05)),
    ]

    for i, (key, title, ylim) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        for exp_data in experiments_with_history:
            exp     = exp_data["exp"]
            history = exp_data["history"]
            if not history or key not in history:
                continue
            epochs = range(1, len(history[key]) + 1)
            ax.plot(epochs, history[key], label=exp["label"],
                    color=exp["color"], linewidth=2)
            if history.get("best_epoch"):
                bp = history["best_epoch"] - 1
                ax.scatter([history["best_epoch"]], [history[key][bp]],
                           color=exp["color"], zorder=5, s=60,
                           edgecolor="black", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key.replace("val_", ""))
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

    # Tabel summary
    ax_sum = fig.add_subplot(gs[3, :])
    ax_sum.axis("off")

    lines = []
    lines.append(f"{'Eksperimen':<28s} {'BestEp':>6s} {'ValLoss':>8s} "
                 f"{'Acc':>6s} {'F1(M)':>6s} {'Rec@Cheat':>10s}")
    lines.append("─" * 80)

    best_recC = -1.0
    winner = None

    for exp_data in experiments_with_history:
        exp = exp_data["exp"]
        h   = exp_data["history"]
        if not h or not h.get("best_epoch"):
            lines.append(f"{exp['label']:<28s} (gagal/no data)")
            continue
        bp = h["best_epoch"] - 1
        recC = h["val_rec_cheat"][bp] if "val_rec_cheat" in h else 0
        lines.append(
            f"{exp['label']:<28s} "
            f"{h['best_epoch']:>6d} "
            f"{h['val_loss'][bp]:>8.4f} "
            f"{h['val_acc'][bp]:>6.3f} "
            f"{h['val_f1'][bp]:>6.3f} "
            f"{recC:>10.4f}"
        )
        # Pemenang ditentukan oleh Recall@Cheat (paling penting untuk task ini)
        if recC > best_recC:
            best_recC = recC
            winner = exp["label"]

    lines.append("")
    if winner:
        lines.append(f"🏆 Best Recall@Cheat: {winner} → Rec@Cheat = {best_recC:.4f}")
        lines.append("   (Rec@Cheat = seberapa baik mendeteksi kelas cheating)")

    summary_text = "\n".join(lines)
    ax_sum.text(0.02, 0.95, summary_text, transform=ax_sum.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5",
                          facecolor="#F5F5F5", edgecolor="#888888"))
    ax_sum.set_title("Ringkasan Perbandingan (di Best Epoch)",
                     fontsize=12, fontweight="bold")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Grafik perbandingan tersimpan: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-root",        default="crop")
    parser.add_argument("--epochs",           type=int,   default=50)
    parser.add_argument("--batch-size",       type=int,   default=32)
    parser.add_argument("--seq-len",          type=int,   default=8)
    parser.add_argument("--stride",           type=int,   default=4)
    parser.add_argument("--hidden-dim",       type=int,   default=128)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--patience",         type=int,   default=10)
    parser.add_argument("--labeling",         default="majority",
                        choices=["any", "majority", "threshold"])
    parser.add_argument("--no-sliding",       action="store_true")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--no-pos-weight",    action="store_true")
    parser.add_argument("--skip-training",    action="store_true")
    parser.add_argument("--output",           default="comparison_results.png")

    args = parser.parse_args()

    if not args.skip_training:
        for exp in EXPERIMENTS:
            if not Path(exp["feature_root"]).exists():
                log.error(f"Folder fitur tidak ada: {exp['feature_root']}")
                log.error(f"Jalankan dulu: python split_features.py")
                return
            ok = run_experiment(exp, args)
            if not ok:
                log.warning(f"Eksperimen {exp['name']} gagal, lanjut...")

    experiments_with_history = []
    for exp in EXPERIMENTS:
        history = load_history(exp["output_dir"])
        experiments_with_history.append({"exp": exp, "history": history})

    plot_comparison(experiments_with_history, args.output)

    # Print summary
    log.info(f"\n{'='*60}")
    log.info("RINGKASAN PERBANDINGAN (Best Epoch)")
    log.info(f"{'='*60}")
    log.info(f"{'Group':<22s} {'BestEp':>6s} {'ValLoss':>8s} {'F1(M)':>6s} "
             f"{'Rec@Cheat':>10s}")
    log.info("-" * 60)
    for exp_data in experiments_with_history:
        exp = exp_data["exp"]
        h   = exp_data["history"]
        if h and h.get("best_epoch"):
            bp = h["best_epoch"] - 1
            recC = h["val_rec_cheat"][bp] if "val_rec_cheat" in h else 0
            log.info(f"{exp['name']:<22s} {h['best_epoch']:>6d} "
                     f"{h['val_loss'][bp]:>8.4f} "
                     f"{h['val_f1'][bp]:>6.3f} "
                     f"{recC:>10.4f}")
        else:
            log.info(f"{exp['name']:<22s} (no data)")


if __name__ == "__main__":
    main()
