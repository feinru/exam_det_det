"""
train.py — Pipeline Training GRU untuk Deteksi Kecurangan Ujian
================================================================
Versi final dengan dukungan:
  - Sliding window (RECOMMENDED untuk dataset imbalance)
  - Auto-detect feature_dim (untuk eksperimen perbandingan fitur)
  - Multiple labeling strategies (any / majority / threshold)
  - Class imbalance: pos_weight + WeightedRandomSampler
  - Early stopping berdasarkan val_loss
  - Macro-averaged metrics (Precision, Recall, F1)
"""

import os
import json
import time
import random
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple, Union
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import lokal
from model import CheatingGRU
from dataset import (
    ExamCheatingDataset,
    ExamCheatingDatasetSliding,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Training Config
# ─────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Path
    feature_root: str = "features"
    crop_root: str    = "crop"
    output_dir: str   = "output"

    # Dataset
    seq_len: int       = 8
    feature_dim: Optional[int] = None  # None = auto-detect dari .npy
    labeling: str      = "majority"    # "any" | "majority" | "threshold"
    threshold: float   = 0.3           # untuk strategy="threshold"
    use_scaler: bool   = True

    # Sliding window (RECOMMENDED)
    use_sliding: bool        = True
    stride: int              = 4
    min_valid_ratio: float   = 0.5

    # Model
    hidden_dim: int     = 128
    num_layers: int     = 2
    fc_dim: int         = 64
    dropout: float      = 0.3
    bidirectional: bool = False
    use_attention: bool = True

    # Training
    epochs: int          = 50
    batch_size: int      = 32
    learning_rate: float = 1e-3
    weight_decay: float  = 1e-4
    num_workers: int     = 0

    # Class imbalance
    use_pos_weight: bool       = True
    use_weighted_sampler: bool = False

    # Early Stopping
    patience: int     = 10
    min_delta: float  = 1e-4

    # LR Scheduler
    lr_factor: float   = 0.5
    lr_patience: int   = 5
    lr_min: float      = 1e-6

    # Lainnya
    seed: int    = 42
    device: str  = "auto"


# ─────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


# ─────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.counter     = 0
        self.best_loss   = float("inf")
        self.best_state  = None
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module) -> bool:
        improved = val_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss   = val_loss
            self.best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter     = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            log.info("Model terbaik dipulihkan dari early stopping.")


# ─────────────────────────────────────────────────────────────────
# Helper: ambil labels dari dataset (samples ATAU windows)
# ─────────────────────────────────────────────────────────────────
def get_labels_from_dataset(dataset) -> list:
    """Helper agar bisa baca label dari kedua tipe dataset."""
    if hasattr(dataset, "windows"):
        return [w[1] for w in dataset.windows]
    else:
        return [s[1] for s in dataset.samples]


# ─────────────────────────────────────────────────────────────────
# Training & Validation
# ─────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, amp_scaler=None):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    all_preds  = []
    all_labels = []

    for features, labels in loader:
        features = features.to(device)
        labels   = labels.float().to(device).unsqueeze(1)

        optimizer.zero_grad()

        if amp_scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(features)
                loss   = criterion(logits, labels)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            logits = model(features)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        preds = (torch.sigmoid(logits.detach()) >= 0.5).long().cpu().numpy().flatten()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy().flatten())
        total_loss += loss.item()
        n_batches  += 1

    avg_loss      = total_loss / n_batches
    all_preds_np  = np.array(all_preds)
    all_labels_np = np.array(all_labels).astype(int)
    acc  = (all_preds_np == all_labels_np).mean()
    prec = precision_score(all_labels_np, all_preds_np, average="macro", zero_division=0)
    rec  = recall_score(all_labels_np, all_preds_np, average="macro", zero_division=0)
    f1   = f1_score(all_labels_np, all_preds_np, average="macro", zero_division=0)
    # Recall khusus kelas cheat (kelas 1)
    rec_cheat = recall_score(all_labels_np, all_preds_np, pos_label=1, zero_division=0)

    return {"loss": avg_loss, "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "rec_cheat": rec_cheat}


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    all_preds  = []
    all_labels = []

    for features, labels in loader:
        features = features.to(device)
        labels   = labels.float().to(device).unsqueeze(1)

        logits = model(features)
        loss   = criterion(logits, labels)

        preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy().flatten()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().flatten())
        total_loss += loss.item()
        n_batches  += 1

    avg_loss      = total_loss / n_batches
    all_preds_np  = np.array(all_preds)
    all_labels_np = np.array(all_labels).astype(int)
    acc  = (all_preds_np == all_labels_np).mean()
    prec = precision_score(all_labels_np, all_preds_np, average="macro", zero_division=0)
    rec  = recall_score(all_labels_np, all_preds_np, average="macro", zero_division=0)
    f1   = f1_score(all_labels_np, all_preds_np, average="macro", zero_division=0)
    rec_cheat = recall_score(all_labels_np, all_preds_np, pos_label=1, zero_division=0)

    return {"loss": avg_loss, "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "rec_cheat": rec_cheat}


# ─────────────────────────────────────────────────────────────────
# Visualisasi
# ─────────────────────────────────────────────────────────────────
def plot_training_history(history: dict, output_path: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    best_ep = history.get("best_epoch")

    fig = plt.figure(figsize=(16, 22))
    fig.suptitle(
        "Training History — Cheating Detection GRU\n(Macro-Averaged + Recall@Cheat)",
        fontsize=16, fontweight="bold"
    )
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.3)

    c_train, c_val, c_best = "#2196F3", "#F44336", "#4CAF50"

    def _add_best_line(ax):
        if best_ep:
            ax.axvline(best_ep, color=c_best, linestyle=":", alpha=0.7,
                       label=f"Best (ep {best_ep})")

    panels = [
        ("loss",       "Loss",                            (0, 0), None),
        ("acc",        "Accuracy",                        (0, 1), (0, 1.05)),
        ("prec",       "Precision (Macro)",               (1, 0), (0, 1.05)),
        ("rec",        "Recall (Macro)",                  (1, 1), (0, 1.05)),
        ("f1",         "F1-Score (Macro)",                (2, 0), (0, 1.05)),
        ("rec_cheat",  "Recall (Kelas Cheat) — KEY",      (2, 1), (0, 1.05)),
    ]

    for key, title, pos, ylim in panels:
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        tkey, vkey = f"train_{key}", f"val_{key}"
        if tkey in history:
            ax.plot(epochs, history[tkey], label="Train", color=c_train, linewidth=2)
        if vkey in history:
            ax.plot(epochs, history[vkey], label="Val", color=c_val,
                    linewidth=2, linestyle="--")
        _add_best_line(ax)
        ax.set_title(f"{title} per Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Learning rate
    ax_lr = fig.add_subplot(gs[3, 0])
    ax_lr.semilogy(epochs, history["lr"], color="#607D8B", linewidth=2)
    ax_lr.set_title("Learning Rate (log scale)")
    ax_lr.set_xlabel("Epoch"); ax_lr.set_ylabel("LR")
    ax_lr.grid(True, alpha=0.3)

    # Best epoch detail
    ax_detail = fig.add_subplot(gs[3, 1])
    if best_ep:
        bp = best_ep - 1
        metrics_short = {
            "Loss": history["val_loss"][bp],
            "Acc":  history["val_acc"][bp],
            "Prec": history["val_prec"][bp],
            "Rec":  history["val_rec"][bp],
            "F1":   history["val_f1"][bp],
            "RecC": history["val_rec_cheat"][bp] if "val_rec_cheat" in history else 0,
        }
        names = list(metrics_short.keys())
        vals  = list(metrics_short.values())
        bars = ax_detail.bar(names, vals, color=[c_val]*len(names), alpha=0.7)
        for bar, val in zip(bars, vals):
            ax_detail.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f"{val:.3f}", ha="center", fontsize=9)
        ax_detail.set_ylim(0, 1.1)
        ax_detail.set_title(f"Val Metrics at Best Epoch ({best_ep})")
        ax_detail.grid(True, alpha=0.3, axis="y")
    else:
        ax_detail.text(0.5, 0.5, "No best epoch", ha="center", va="center",
                       transform=ax_detail.transAxes)
        ax_detail.axis("off")

    # Summary text
    ax_sum = fig.add_subplot(gs[4, :])
    ax_sum.axis("off")
    if best_ep:
        bp = best_ep - 1
        rec_cheat_str = (f"{history['val_rec_cheat'][bp]:.4f}"
                         if "val_rec_cheat" in history else "n/a")
        summary_text = "\n".join([
            f"Best Epoch: {best_ep}",
            f"  Val Loss     : {history['val_loss'][bp]:.6f}",
            f"  Val Accuracy : {history['val_acc'][bp]:.4f}",
            f"  Val Prec(M)  : {history['val_prec'][bp]:.4f}",
            f"  Val Rec(M)   : {history['val_rec'][bp]:.4f}",
            f"  Val F1(M)    : {history['val_f1'][bp]:.4f}",
            f"  Val Rec@Cheat: {rec_cheat_str}  ← seberapa baik mendeteksi cheating",
            f"  Learning Rate: {history['lr'][bp]:.2e}",
        ])
    else:
        summary_text = "No best epoch recorded."
    ax_sum.text(0.05, 0.95, summary_text, transform=ax_sum.transAxes,
                fontsize=11, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5",
                          facecolor="#F5F5F5", edgecolor="#CCCCCC"))
    ax_sum.set_title("Training Summary", fontsize=12, fontweight="bold")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Grafik training tersimpan: {output_path}")


# ─────────────────────────────────────────────────────────────────
# Class imbalance helpers
# ─────────────────────────────────────────────────────────────────
def compute_pos_weight(dataset) -> torch.Tensor:
    labels = [l for l in get_labels_from_dataset(dataset) if l >= 0]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0:
        log.warning("Tidak ada sampel positif (cheat) di training!")
        return torch.tensor([1.0])
    pw = n_neg / n_pos
    log.info(f"pos_weight = {pw:.3f}  (n_neg={n_neg}, n_pos={n_pos})")
    return torch.tensor([pw], dtype=torch.float32)


def build_weighted_sampler(dataset) -> WeightedRandomSampler:
    labels = get_labels_from_dataset(dataset)
    n_pos  = sum(l == 1 for l in labels)
    n_neg  = sum(l == 0 for l in labels)
    n_total = len(labels)
    if n_pos == 0 or n_neg == 0:
        log.warning("Salah satu kelas kosong, WeightedSampler tidak efektif.")
        weights = [1.0] * n_total
    else:
        w_pos = n_total / (2 * n_pos)
        w_neg = n_total / (2 * n_neg)
        weights = [w_pos if l == 1 else w_neg for l in labels]
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float64),
        num_samples=n_total, replacement=True,
    )


# ─────────────────────────────────────────────────────────────────
# Main Training
# ─────────────────────────────────────────────────────────────────
def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Device : {device}")
    log.info(f"Output : {out_dir.resolve()}")
    log.info(f"Sliding window : {cfg.use_sliding} "
             f"(stride={cfg.stride}, window={cfg.seq_len})")

    # ── DataLoaders ──────────────────────────────────────────────
    log.info("Memuat dataset...")

    common_args = dict(
        feature_root=cfg.feature_root,
        crop_root=cfg.crop_root,
        feature_dim=cfg.feature_dim,
        labeling=cfg.labeling,
        threshold=cfg.threshold,
        use_scaler=cfg.use_scaler,
    )

    if cfg.use_sliding:
        train_ds = ExamCheatingDatasetSliding(
            split="train", window_size=cfg.seq_len, stride=cfg.stride,
            min_valid_ratio=cfg.min_valid_ratio, augment=True, **common_args,
        )
        fitted_scaler = train_ds.scaler if cfg.use_scaler else None
        actual_dim = train_ds.feature_dim
        common_args["feature_dim"] = actual_dim
        common_args["scaler"]      = fitted_scaler

        val_ds = ExamCheatingDatasetSliding(
            split="valid", window_size=cfg.seq_len, stride=cfg.stride,
            min_valid_ratio=cfg.min_valid_ratio, **common_args,
        )
    else:
        train_ds = ExamCheatingDataset(
            split="train", seq_len=cfg.seq_len, augment=True, **common_args,
        )
        fitted_scaler = train_ds.scaler if cfg.use_scaler else None
        actual_dim = train_ds.feature_dim
        common_args["feature_dim"] = actual_dim
        common_args["scaler"]      = fitted_scaler

        val_ds = ExamCheatingDataset(
            split="valid", seq_len=cfg.seq_len, **common_args,
        )

    # Sampler
    sampler = None
    if cfg.use_weighted_sampler:
        sampler = build_weighted_sampler(train_ds)
        log.info("WeightedRandomSampler aktif.")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    log.info(f"Train: {len(train_ds)} sampel | Val: {len(val_ds)} sampel")
    log.info(f"Feature dim (auto-detected): {actual_dim}")

    # ── Model ────────────────────────────────────────────────────
    model = CheatingGRU(
        input_dim=actual_dim,       # ← PENTING: pakai dim yang ke-detect
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        fc_dim=cfg.fc_dim,
        dropout=cfg.dropout,
        bidirectional=cfg.bidirectional,
        use_attention=cfg.use_attention,
    ).to(device)
    model.summary()

    # ── Loss ─────────────────────────────────────────────────────
    pos_weight = compute_pos_weight(train_ds).to(device) if cfg.use_pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimizer & Scheduler ────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate,
                     weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.lr_min,
    )

    # ── Early Stopping ───────────────────────────────────────────
    early_stop = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)
    amp_scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Training Loop ────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [],  "val_acc": [],
        "train_prec": [], "val_prec": [],
        "train_rec": [],  "val_rec": [],
        "train_f1": [],   "val_f1": [],
        "train_rec_cheat": [], "val_rec_cheat": [],
        "lr": [], "best_epoch": None,
    }

    log.info(f"\n{'=' * 55}\nMULAI TRAINING\n{'=' * 55}")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tm = train_one_epoch(model, train_loader, optimizer, criterion, device, amp_scaler)
        vm = validate_one_epoch(model, val_loader, criterion, device)

        scheduler.step(vm["loss"])
        lr = optimizer.param_groups[0]["lr"]

        for key in ["loss", "acc", "prec", "rec", "f1", "rec_cheat"]:
            history[f"train_{key}"].append(tm[key])
            history[f"val_{key}"].append(vm[key])
        history["lr"].append(lr)

        improved = early_stop.step(vm["loss"], model)
        if improved:
            history["best_epoch"] = epoch
        marker = " ★" if improved else f" (no improve {early_stop.counter}/{cfg.patience})"

        log.info(
            f"Ep {epoch:03d}/{cfg.epochs} | "
            f"Loss={tm['loss']:.3f}/{vm['loss']:.3f} | "
            f"Acc={tm['acc']:.3f}/{vm['acc']:.3f} | "
            f"F1(M)={vm['f1']:.3f} | RecCheat={vm['rec_cheat']:.3f} | "
            f"{time.time()-t0:.1f}s{marker}"
        )

        if early_stop.should_stop:
            log.info(f"\nEarly stop di epoch {epoch}. "
                     f"Best val_loss={early_stop.best_loss:.4f} "
                     f"di epoch {history['best_epoch']}")
            break

    # ── Save ─────────────────────────────────────────────────────
    early_stop.restore_best(model)
    model_path = out_dir / "best_model.pth"
    cfg_dict = asdict(cfg)
    cfg_dict["feature_dim"] = actual_dim  # ← simpan dim aktual untuk inference
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg_dict,
        "best_val_loss": early_stop.best_loss,
        "best_epoch":    history["best_epoch"],
        "history":       history,
    }, str(model_path))
    log.info(f"\nModel terbaik tersimpan: {model_path}")

    plot_path = out_dir / "training_history.png"
    plot_training_history(history, str(plot_path))

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info(f"\n{'=' * 55}\nRINGKASAN TRAINING\n{'=' * 55}")
    if history["best_epoch"]:
        bp = history["best_epoch"] - 1
        log.info(f"Best epoch        : {history['best_epoch']}")
        log.info(f"Best val loss     : {early_stop.best_loss:.4f}")
        log.info(f"Best val acc      : {history['val_acc'][bp]:.4f}")
        log.info(f"Best val F1(M)    : {history['val_f1'][bp]:.4f}")
        log.info(f"Best val Rec@Cheat: {history['val_rec_cheat'][bp]:.4f}")
    log.info(f"Model tersimpan   : {model_path}")
    log.info(f"Grafik            : {plot_path}")

    return model, history


# ─────────────────────────────────────────────────────────────────
# Helper untuk Fase 3
# ─────────────────────────────────────────────────────────────────
def load_best_model(checkpoint_path: str, device: str = "auto"):
    _device = get_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    cfg_dict = checkpoint["config"]

    model = CheatingGRU(
        input_dim=cfg_dict.get("feature_dim", 38),
        hidden_dim=cfg_dict["hidden_dim"],
        num_layers=cfg_dict["num_layers"],
        fc_dim=cfg_dict["fc_dim"],
        dropout=cfg_dict["dropout"],
        bidirectional=cfg_dict["bidirectional"],
        use_attention=cfg_dict["use_attention"],
    ).to(_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    log.info(f"Model dimuat dari {checkpoint_path}")
    return model, cfg_dict


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fase 2: GRU Training")

    # Path
    parser.add_argument("--feature-root", default="features")
    parser.add_argument("--crop-root",    default="crop")
    parser.add_argument("--output-dir",   default="output")

    # Dataset
    parser.add_argument("--seq-len",     type=int,   default=8)
    parser.add_argument("--feature-dim", type=int,   default=None,
                        help="None untuk auto-detect")
    parser.add_argument("--labeling",    default="majority",
                        choices=["any", "majority", "threshold"])
    parser.add_argument("--threshold",   type=float, default=0.3)
    parser.add_argument("--no-scaler",   action="store_true")

    # Sliding window
    parser.add_argument("--no-sliding",       action="store_true",
                        help="Disable sliding window (pakai per-siswa)")
    parser.add_argument("--stride",           type=int,   default=4)
    parser.add_argument("--min-valid-ratio",  type=float, default=0.5)

    # Model
    parser.add_argument("--hidden-dim",    type=int,   default=128)
    parser.add_argument("--num-layers",    type=int,   default=2)
    parser.add_argument("--fc-dim",        type=int,   default=64)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--no-attention",  action="store_true")

    # Training
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # Imbalance
    parser.add_argument("--no-pos-weight",    action="store_true")
    parser.add_argument("--weighted-sampler", action="store_true")

    # Early stop
    parser.add_argument("--patience",  type=int,   default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)

    # Lain
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--num-workers", type=int,   default=0)

    args = parser.parse_args()

    cfg = TrainConfig(
        feature_root=args.feature_root,
        crop_root=args.crop_root,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        feature_dim=args.feature_dim,
        labeling=args.labeling,
        threshold=args.threshold,
        use_scaler=not args.no_scaler,
        use_sliding=not args.no_sliding,
        stride=args.stride,
        min_valid_ratio=args.min_valid_ratio,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        fc_dim=args.fc_dim,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        use_attention=not args.no_attention,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        use_pos_weight=not args.no_pos_weight,
        use_weighted_sampler=args.weighted_sampler,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
    )

    train(cfg)
