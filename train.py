"""
train.py — Pipeline Training GRU untuk Deteksi Kecurangan Ujian
================================================================
[v4 + multi-variant]

Perubahan utama vs train.py lama:
- Tambah flag --feature-variant {coord,geom,full} → otomatis pilih
  feature root, output dir, dan input_dim model.
- feature_dim model di-detect dari variant_meta.json (tidak lagi fixed 38).
- Output disimpan ke {output_dir}/{variant}/ untuk pemisahan eksperimen.
"""

import os
import json
import time
import random
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, field, asdict

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

from model import CheatingGRU
from dataset import ExamCheatingDataset, detect_feature_dim

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# Mapping varian → default feature root
VARIANT_DEFAULT_ROOTS = {
    "coord": "features_coord",
    "geom":  "features_geom",
    "full":  "features_full",
}

VARIANT_DIMS = {"coord": 23, "geom": 28, "full": 38}


# ─────────────────────────────────────────────────────────────────
# Konfigurasi Training
# ─────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Path
    feature_root: str = "features_full"
    crop_root: str = "crop"
    output_dir: str = "output"
    feature_variant: str = "full"  # ← BARU

    # Dataset
    seq_len: int = 8
    labeling: str = "any"
    use_scaler: bool = True

    # Model (input_dim akan di-override dari feature root)
    input_dim: int = 38
    hidden_dim: int = 128
    num_layers: int = 2
    fc_dim: int = 64
    dropout: float = 0.3
    bidirectional: bool = False
    use_attention: bool = True

    # Training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0

    # Class imbalance
    use_pos_weight: bool = True
    use_weighted_sampler: bool = False

    # Early Stopping
    patience: int = 10
    min_delta: float = 1e-4

    # LR Scheduler
    lr_factor: float = 0.5
    lr_patience: int = 5
    lr_min: float = 1e-6

    # Lain
    seed: int = 42
    device: str = "auto"


# ─────────────────────────────────────────────────────────────────
# Utilitas
# ─────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_state = None
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module) -> bool:
        improved = val_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss = val_loss
            self.best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            self.counter = 0
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
# Training & Validation Step
# ─────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler_amp=None):
    model.train()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.float().to(device).unsqueeze(1)

        optimizer.zero_grad()
        if scaler_amp is not None:
            with torch.cuda.amp.autocast():
                logits = model(features)
                loss = criterion(logits, labels)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        preds = (torch.sigmoid(logits.detach()) >= 0.5).long().cpu().numpy().flatten()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy().flatten())

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels).astype(int)
    acc = (all_preds_np == all_labels_np).mean()
    prec = precision_score(all_labels_np, all_preds_np, average="macro", zero_division=0)
    rec = recall_score(all_labels_np, all_preds_np, average="macro", zero_division=0)
    f1 = f1_score(all_labels_np, all_preds_np, average="macro", zero_division=0)

    return {"loss": avg_loss, "acc": acc, "prec": prec, "rec": rec, "f1": f1}


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.float().to(device).unsqueeze(1)

        logits = model(features)
        loss = criterion(logits, labels)

        preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy().flatten()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().flatten())

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels).astype(int)
    acc = (all_preds_np == all_labels_np).mean()
    prec = precision_score(all_labels_np, all_preds_np, average="macro", zero_division=0)
    rec = recall_score(all_labels_np, all_preds_np, average="macro", zero_division=0)
    f1 = f1_score(all_labels_np, all_preds_np, average="macro", zero_division=0)

    return {"loss": avg_loss, "acc": acc, "prec": prec, "rec": rec, "f1": f1}


# ─────────────────────────────────────────────────────────────────
# Visualisasi
# ─────────────────────────────────────────────────────────────────
def plot_training_history(history: dict, output_path: str, title_suffix: str = ""):
    epochs = range(1, len(history["train_loss"]) + 1)
    best_ep = history.get("best_epoch")

    fig = plt.figure(figsize=(16, 20))
    title = "Training History — Cheating Detection GRU"
    if title_suffix:
        title += f" [{title_suffix}]"
    fig.suptitle(title + "\n(Macro-Averaged Metrics)",
                 fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    c_train = "#2196F3"
    c_val = "#F44336"
    c_best = "#4CAF50"

    def _add_best_line(ax):
        if best_ep:
            ax.axvline(best_ep, color=c_best, linestyle=":", alpha=0.7,
                       label=f"Best (ep {best_ep})")

    # Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["train_loss"], label="Train", color=c_train, linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val", color=c_val, linewidth=2, linestyle="--")
    _add_best_line(ax1)
    ax1.set_title("Loss per Epoch"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history["train_acc"], label="Train", color=c_train, linewidth=2)
    ax2.plot(epochs, history["val_acc"], label="Val", color=c_val, linewidth=2, linestyle="--")
    _add_best_line(ax2)
    ax2.set_title("Accuracy per Epoch"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # Precision
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, history["train_prec"], label="Train", color=c_train, linewidth=2)
    ax3.plot(epochs, history["val_prec"], label="Val", color=c_val, linewidth=2, linestyle="--")
    _add_best_line(ax3)
    ax3.set_title("Precision (Macro) per Epoch"); ax3.set_xlabel("Epoch"); ax3.set_ylabel("Precision")
    ax3.set_ylim(0, 1.05); ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # Recall
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, history["train_rec"], label="Train", color=c_train, linewidth=2)
    ax4.plot(epochs, history["val_rec"], label="Val", color=c_val, linewidth=2, linestyle="--")
    _add_best_line(ax4)
    ax4.set_title("Recall (Macro) per Epoch"); ax4.set_xlabel("Epoch"); ax4.set_ylabel("Recall")
    ax4.set_ylim(0, 1.05); ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    # F1
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(epochs, history["train_f1"], label="Train", color=c_train, linewidth=2)
    ax5.plot(epochs, history["val_f1"], label="Val", color=c_val, linewidth=2, linestyle="--")
    _add_best_line(ax5)
    ax5.set_title("F1-Score (Macro) per Epoch"); ax5.set_xlabel("Epoch"); ax5.set_ylabel("F1")
    ax5.set_ylim(0, 1.05); ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

    # LR
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.semilogy(epochs, history["lr"], color="#607D8B", linewidth=2, label="LR")
    ax6.set_title("Learning Rate (log scale)"); ax6.set_xlabel("Epoch"); ax6.set_ylabel("LR")
    ax6.grid(True, alpha=0.3)

    # Summary
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis("off")
    if best_ep:
        bp = best_ep - 1
        summary_lines = [
            f"Best Epoch: {best_ep}",
            f"  Val Loss : {history['val_loss'][bp]:.6f}",
            f"  Val Acc  : {history['val_acc'][bp]:.4f}",
            f"  Val Prec : {history['val_prec'][bp]:.4f} (macro)",
            f"  Val Rec  : {history['val_rec'][bp]:.4f} (macro)",
            f"  Val F1   : {history['val_f1'][bp]:.4f} (macro)",
            f"  LR       : {history['lr'][bp]:.2e}",
            "",
            f"Final Epoch: {len(history['train_loss'])}",
            f"  Val Loss : {history['val_loss'][-1]:.6f}",
            f"  Val Acc  : {history['val_acc'][-1]:.4f}",
            f"  Val F1   : {history['val_f1'][-1]:.4f} (macro)",
        ]
        summary_text = "\n".join(summary_lines)
    else:
        summary_text = "No best epoch recorded."

    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#CCCCCC"))
    ax7.set_title("Training Summary", fontsize=12, fontweight="bold")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Grafik training tersimpan: {output_path}")


# ─────────────────────────────────────────────────────────────────
# Pos-weight & Sampler
# ─────────────────────────────────────────────────────────────────
def compute_pos_weight(dataset: ExamCheatingDataset) -> torch.Tensor:
    labels = [s[1] for s in dataset.samples if s[1] >= 0]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        log.warning("Tidak ada sampel positif (cheating) di training set!")
        return torch.tensor([1.0])
    pw = n_neg / n_pos
    log.info(f"pos_weight = {pw:.3f} (n_neg={n_neg}, n_pos={n_pos})")
    return torch.tensor([pw], dtype=torch.float32)


def build_weighted_sampler(dataset: ExamCheatingDataset) -> WeightedRandomSampler:
    labels = [s[1] for s in dataset.samples]
    n_pos = sum(l == 1 for l in labels)
    n_neg = sum(l == 0 for l in labels)
    n_total = len(labels)
    if n_pos == 0 or n_neg == 0:
        log.warning("Salah satu kelas kosong, WeightedSampler tidak efektif.")
        weights = [1.0] * n_total
    else:
        w_pos = n_total / (2 * n_pos)
        w_neg = n_total / (2 * n_neg)
        weights = [w_pos if l == 1 else w_neg for l in labels]

    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float64),
        num_samples=n_total,
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────────────────────────
def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect feature_dim dari variant_meta.json di feature_root
    detected_dim = detect_feature_dim(Path(cfg.feature_root))
    if detected_dim != cfg.input_dim:
        log.warning(f"input_dim cfg={cfg.input_dim} dioverride oleh detected_dim={detected_dim} "
                    f"dari {cfg.feature_root}/variant_meta.json")
        cfg.input_dim = detected_dim

    log.info(f"Variant      : {cfg.feature_variant}")
    log.info(f"Feature root : {cfg.feature_root} (dim={cfg.input_dim})")
    log.info(f"Device       : {device}")
    log.info(f"Output       : {out_dir.resolve()}")

    # ── DataLoaders ──────────────────────────────────────────────
    log.info("Memuat dataset...")
    train_ds = ExamCheatingDataset(
        feature_root=cfg.feature_root,
        crop_root=cfg.crop_root,
        split="train",
        seq_len=cfg.seq_len,
        feature_dim=cfg.input_dim,
        labeling=cfg.labeling,
        use_scaler=cfg.use_scaler,
        augment=True,
    )

    val_ds = ExamCheatingDataset(
        feature_root=cfg.feature_root,
        crop_root=cfg.crop_root,
        split="valid",
        seq_len=cfg.seq_len,
        feature_dim=cfg.input_dim,
        labeling=cfg.labeling,
        use_scaler=cfg.use_scaler,
        scaler=train_ds.scaler,
    )

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

    # ── Model ────────────────────────────────────────────────────
    model = CheatingGRU(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        fc_dim=cfg.fc_dim,
        dropout=cfg.dropout,
        bidirectional=cfg.bidirectional,
        use_attention=cfg.use_attention,
    ).to(device)
    if hasattr(model, "summary"):
        model.summary()

    # ── Loss ─────────────────────────────────────────────────────
    if cfg.use_pos_weight:
        pos_weight = compute_pos_weight(train_ds).to(device)
    else:
        pos_weight = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.lr_min,
    )
    early_stop = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)
    amp_scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_prec": [], "train_rec": [], "train_f1": [],
        "val_prec": [], "val_rec": [], "val_f1": [],
        "lr": [], "best_epoch": None,
    }

    log.info(f"\n{'=' * 55}")
    log.info(f"MULAI TRAINING (variant={cfg.feature_variant})")
    log.info(f"{'=' * 55}")

    for epoch in range(1, cfg.epochs + 1):
        t_start = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, amp_scaler)
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        val_loss = val_metrics["loss"]
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])
        history["train_prec"].append(train_metrics["prec"])
        history["train_rec"].append(train_metrics["rec"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_prec"].append(val_metrics["prec"])
        history["val_rec"].append(val_metrics["rec"])
        history["val_f1"].append(val_metrics["f1"])
        history["lr"].append(current_lr)

        elapsed = time.time() - t_start
        improved = early_stop.step(val_loss, model)
        marker = " ★" if improved else f" (no improve {early_stop.counter}/{cfg.patience})"

        log.info(
            f"Ep {epoch:03d}/{cfg.epochs} | "
            f"Loss={train_metrics['loss']:.3f}/{val_loss:.3f} | "
            f"Acc={train_metrics['acc']:.3f}/{val_metrics['acc']:.3f} | "
            f"P={val_metrics['prec']:.3f} R={val_metrics['rec']:.3f} F1={val_metrics['f1']:.3f} | "
            f"LR={current_lr:.2e} | {elapsed:.1f}s{marker}"
        )

        if improved:
            history["best_epoch"] = epoch

        if early_stop.should_stop:
            log.info(f"\nEarly stopping di epoch {epoch}. "
                     f"Best val_loss={early_stop.best_loss:.4f} di epoch {history['best_epoch']}")
            break

    early_stop.restore_best(model)
    model_path = out_dir / "best_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "best_val_loss": early_stop.best_loss,
            "best_epoch": history["best_epoch"],
            "history": history,
        },
        str(model_path),
    )
    log.info(f"\nModel terbaik tersimpan: {model_path}")

    plot_path = out_dir / "training_history.png"
    plot_training_history(history, str(plot_path), title_suffix=f"variant={cfg.feature_variant}")

    history_path = out_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Final summary metrics to a flat JSON for easy comparison
    bp = history["best_epoch"] - 1 if history["best_epoch"] else -1
    summary = {
        "variant": cfg.feature_variant,
        "input_dim": cfg.input_dim,
        "best_epoch": history["best_epoch"],
        "best_val_loss": float(early_stop.best_loss),
        "best_val_acc": float(history["val_acc"][bp]) if bp >= 0 else None,
        "best_val_prec": float(history["val_prec"][bp]) if bp >= 0 else None,
        "best_val_rec": float(history["val_rec"][bp]) if bp >= 0 else None,
        "best_val_f1": float(history["val_f1"][bp]) if bp >= 0 else None,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"\n{'=' * 55}")
    log.info("RINGKASAN TRAINING")
    log.info(f"{'=' * 55}")
    log.info(f"Variant         : {cfg.feature_variant}")
    log.info(f"Best epoch      : {history['best_epoch']}")
    log.info(f"Best val loss   : {early_stop.best_loss:.4f}")
    if bp >= 0:
        log.info(f"Best val acc    : {history['val_acc'][bp]:.4f}")
        log.info(f"Best val F1     : {history['val_f1'][bp]:.4f}")
    log.info(f"Model           : {model_path}")
    log.info(f"Grafik          : {plot_path}")

    return model, history, summary


# ─────────────────────────────────────────────────────────────────
# Load best model (untuk Fase 3)
# ─────────────────────────────────────────────────────────────────
def load_best_model(checkpoint_path: str, device: str = "auto"):
    _device = get_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    cfg_dict = checkpoint["config"]

    input_dim = cfg_dict.get("input_dim", 38)

    model = CheatingGRU(
        input_dim=input_dim,
        hidden_dim=cfg_dict["hidden_dim"],
        num_layers=cfg_dict["num_layers"],
        fc_dim=cfg_dict["fc_dim"],
        dropout=cfg_dict["dropout"],
        bidirectional=cfg_dict["bidirectional"],
        use_attention=cfg_dict["use_attention"],
    ).to(_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    log.info(
        f"Model dimuat dari {checkpoint_path} "
        f"(variant={cfg_dict.get('feature_variant','?')}, "
        f"input_dim={input_dim}, "
        f"best_epoch={checkpoint['best_epoch']}, "
        f"val_loss={checkpoint['best_val_loss']:.4f})"
    )
    return model, cfg_dict


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training GRU — multi-variant")

    # ── Varian fitur (BARU) ──
    parser.add_argument("--feature-variant", choices=list(VARIANT_DIMS.keys()),
                        default="full",
                        help="Varian fitur: coord (23) | geom (28) | full (38)")

    # Path
    parser.add_argument("--feature-root", default=None,
                        help="Override default. Default: features_<variant>")
    parser.add_argument("--output-dir", default=None,
                        help="Override default. Default: output/<variant>")
    parser.add_argument("--crop-root", default="crop")

    # Model
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--fc-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true", default=None)
    parser.add_argument("--no-bidirectional", action="store_true", default=None)
    parser.add_argument("--no-attention", action="store_true")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seq-len", type=int, default=8)

    # Imbalance
    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument("--weighted-sampler", action="store_true")

    # Early stopping
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--labeling", default="any", choices=["any", "majority"])
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    # Resolve default paths
    variant = args.feature_variant
    feature_root = args.feature_root or VARIANT_DEFAULT_ROOTS[variant]
    output_dir = args.output_dir or f"output/{variant}"

    cfg = TrainConfig(
        feature_root=feature_root,
        crop_root=args.crop_root,
        output_dir=output_dir,
        feature_variant=variant,
        seq_len=args.seq_len,
        labeling=args.labeling,
        input_dim=VARIANT_DIMS[variant],  # akan di-override oleh detect_feature_dim
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        fc_dim=args.fc_dim,
        dropout=args.dropout,
        bidirectional=(args.bidirectional if args.bidirectional is not None
                       else (False if args.no_bidirectional else TrainConfig.bidirectional)),
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
