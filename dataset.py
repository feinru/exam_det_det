"""
dataset.py v4 — PyTorch Dataset dengan Label Per-Siswa yang BENAR
==================================================================

[v4 + multi-variant]
Membaca labels_per_student.json dari hasil Fase 0 v3,
dan auto-detect FEATURE_DIM dari variant_meta.json (atau dari npy pertama).

Mendukung 3 varian fitur untuk eksperimen pembanding:
  - coord (23 dim)
  - geom  (28 dim)
  - full  (38 dim)
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Literal
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────────────
SEQ_LEN = 8  # default; bisa di-override per call
# FEATURE_DIM tidak lagi konstan global — auto-detect.
DEFAULT_FEATURE_DIM = 38  # fallback untuk kompatibilitas

# Untuk augmentor: index velocity & visibility berbeda per varian
VARIANT_INDICES = {
    23: {  # coord
        "vis": (21, 23),
        "velocity": None,
    },
    28: {  # geom
        "vis": (26, 28),
        "velocity": None,
    },
    38: {  # full
        "vis": (26, 28),
        "velocity": (28, 38),
    },
}
# ─────────────────────────────────────────────────────────────────


def detect_feature_dim(feature_root: Path) -> int:
    """Cek dari variant_meta.json, kalau tidak ada baca npy pertama."""
    meta_path = feature_root / "variant_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        return int(meta["feature_dim"])

    # Fallback: cari npy pertama
    for split in ["train", "valid", "test"]:
        split_dir = feature_root / split
        if not split_dir.exists():
            continue
        for video_dir in split_dir.iterdir():
            if not video_dir.is_dir():
                continue
            for npy in video_dir.glob("*.npy"):
                arr = np.load(str(npy))
                return int(arr.shape[-1])

    return DEFAULT_FEATURE_DIM


# ─────────────────────────────────────────────────────────────────
# Data Augmentation
# ─────────────────────────────────────────────────────────────────
class TemporalAugmentor:
    """Augmentasi sequence temporal untuk dataset kecil.
    Sadar varian — flipping velocity dan masking visibility disesuaikan.
    """
    def __init__(self, feature_dim: int = 38,
                 p_noise=0.3, p_flip=0.1, p_mask=0.2,
                 p_jitter=0.2, noise_std=0.01, mask_ratio=0.10):
        self.feature_dim = feature_dim
        self.indices = VARIANT_INDICES.get(feature_dim, VARIANT_INDICES[38])
        self.p_noise = p_noise
        self.p_flip = p_flip
        self.p_mask = p_mask
        self.p_jitter = p_jitter
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio

    def __call__(self, feat: np.ndarray) -> np.ndarray:
        feat = feat.copy()

        if np.random.random() < self.p_noise:
            feat = feat + np.random.normal(0, self.noise_std, feat.shape).astype(np.float32)

        if np.random.random() < self.p_flip:
            feat = feat[::-1].copy()
            # Flip velocity sign jika ada
            vel_idx = self.indices["velocity"]
            if vel_idx is not None:
                feat[:, vel_idx[0]:vel_idx[1]] = -feat[:, vel_idx[0]:vel_idx[1]]

        if np.random.random() < self.p_mask:
            n_mask = max(1, int(feat.shape[1] * self.mask_ratio))
            # Jangan mask kolom visibility — itu sinyal eksplisit
            vis_idx = self.indices["vis"]
            vis_cols = set(range(vis_idx[0], vis_idx[1]))
            candidates = [i for i in range(feat.shape[1]) if i not in vis_cols]
            mask_cols = np.random.choice(candidates, size=min(n_mask, len(candidates)),
                                          replace=False)
            feat[:, mask_cols] = 0.0

        if np.random.random() < self.p_jitter:
            shift = np.random.randint(-2, 3)
            if shift != 0:
                shifted = np.zeros_like(feat)
                T = feat.shape[0]
                if shift > 0:
                    shifted[shift:] = feat[:T - shift]
                else:
                    shifted[:T + shift] = feat[-shift:]
                feat = shifted

        return feat


# ─────────────────────────────────────────────────────────────────
# Per-student label loading
# ─────────────────────────────────────────────────────────────────
def load_per_student_labels(
    crop_video_dir: Path,
    student_id: str,
    labeling: Literal["any", "majority"] = "any",
) -> int:
    """Baca label per-siswa dari labels_per_student.json.

    Return: 0 = not_cheating, 1 = cheating, -1 = tidak ada label.
    """
    labels_json_path = crop_video_dir / "labels_per_student.json"
    if not labels_json_path.exists():
        return -1

    with open(labels_json_path, "r") as f:
        all_labels = json.load(f)

    student_labels = all_labels.get(student_id, [])
    if not student_labels:
        return -1

    valid_labels = [c for c in student_labels if c >= 0]
    if not valid_labels:
        return -1

    # 0 = cheating, 1 = not_cheating di anotasi YOLO Fikar
    n_cheating = sum(1 for c in valid_labels if c == 0)
    n_total = len(valid_labels)

    if labeling == "any":
        return 1 if n_cheating > 0 else 0
    else:  # majority
        return 1 if (n_cheating / n_total) > 0.5 else 0


# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────
class ExamCheatingDataset(Dataset):
    """Tiap sampel = 1 siswa dengan label PER-SISWA.

    Penggunaan:
        train_ds = ExamCheatingDataset(
            feature_root="features_full",   # ← pilih varian via folder
            crop_root="crop",
            split="train",
        )
        # feature_dim auto-detect dari variant_meta.json
    """

    def __init__(
        self,
        feature_root: str,
        crop_root: str,
        split: str = "train",
        seq_len: int = SEQ_LEN,
        feature_dim: Optional[int] = None,
        labeling: Literal["any", "majority"] = "any",
        use_scaler: bool = False,
        scaler=None,
        augment: bool = False,
        verbose: bool = True,
    ):
        self.seq_len = seq_len
        self.split = split
        self.use_scaler = use_scaler

        feature_root_path = Path(feature_root)

        # Auto-detect feature_dim kalau tidak dikasih
        if feature_dim is None:
            feature_dim = detect_feature_dim(feature_root_path)
        self.feature_dim = feature_dim

        # Augmentasi HANYA saat training
        self.augmentor = (
            TemporalAugmentor(feature_dim=feature_dim)
            if (augment and split == "train") else None
        )

        feature_split_dir = feature_root_path / split
        crop_split_dir = Path(crop_root) / split

        self.samples = []  # list of (npy_path, label)
        for video_dir in sorted(feature_split_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name
            crop_video_dir = crop_split_dir / video_id

            for npy_file in sorted(video_dir.glob("*.npy")):
                student_id = npy_file.stem
                label = load_per_student_labels(
                    crop_video_dir, student_id, labeling
                )
                self.samples.append((npy_file, label))

        valid_samples = [s for s in self.samples if s[1] >= 0]
        n_dropped = len(self.samples) - len(valid_samples)
        self.samples = valid_samples

        if verbose:
            labels = [s[1] for s in self.samples]
            dist = Counter(labels)
            print(f"[Dataset/{split}] feature_dim={feature_dim} "
                  f"| {len(self.samples)} sampel valid "
                  f"({n_dropped} di-drop)")
            print(f"[Dataset/{split}] distribusi: "
                  f"not_cheat(0)={dist.get(0,0)}, cheat(1)={dist.get(1,0)}")

        # Scaler
        self.scaler = None
        if use_scaler:
            if scaler is not None:
                self.scaler = scaler
            elif split == "train":
                self.scaler = self._fit_scaler()
            else:
                raise ValueError("use_scaler=True tapi split bukan train, scaler kosong.")

    def _fit_scaler(self):
        from sklearn.preprocessing import StandardScaler
        all_data = []
        for npy_path, _ in self.samples:
            all_data.append(np.load(str(npy_path)))
        all_data = np.concatenate(all_data, axis=0)
        scaler = StandardScaler()
        scaler.fit(all_data)
        return scaler

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        npy_path, label = self.samples[idx]
        feat = np.load(str(npy_path)).astype(np.float32)

        if feat.shape[0] != self.seq_len:
            if feat.shape[0] > self.seq_len:
                feat = feat[:self.seq_len]
            else:
                pad = np.zeros((self.seq_len - feat.shape[0], self.feature_dim),
                               dtype=np.float32)
                feat = np.concatenate([feat, pad], axis=0)

        if self.scaler is not None:
            feat = self.scaler.transform(feat).astype(np.float32)

        if self.augmentor is not None:
            feat = self.augmentor(feat)

        return (
            torch.from_numpy(feat),
            torch.tensor(label, dtype=torch.long),
        )

    def get_class_weights(self) -> torch.Tensor:
        labels = [s[1] for s in self.samples]
        n_total = len(labels)
        n_cheating = sum(labels)
        n_not = n_total - n_cheating

        if n_cheating == 0 or n_not == 0:
            return torch.ones(2)

        w_not = n_total / (2 * n_not)
        w_cheating = n_total / (2 * n_cheating)
        return torch.tensor([w_not, w_cheating], dtype=torch.float32)


def build_dataloaders(
    feature_root: str,
    crop_root: str,
    seq_len: int = SEQ_LEN,
    feature_dim: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    use_scaler: bool = False,
    labeling: str = "any",
    pin_memory: bool = True,
):
    if feature_dim is None:
        feature_dim = detect_feature_dim(Path(feature_root))

    train_ds = ExamCheatingDataset(
        feature_root=feature_root, crop_root=crop_root,
        split="train", seq_len=seq_len, feature_dim=feature_dim,
        labeling=labeling, use_scaler=use_scaler,
    )

    fitted_scaler = train_ds.scaler if use_scaler else None

    valid_ds = ExamCheatingDataset(
        feature_root=feature_root, crop_root=crop_root,
        split="valid", seq_len=seq_len, feature_dim=feature_dim,
        labeling=labeling, use_scaler=use_scaler, scaler=fitted_scaler,
    )

    test_ds = ExamCheatingDataset(
        feature_root=feature_root, crop_root=crop_root,
        split="test", seq_len=seq_len, feature_dim=feature_dim,
        labeling=labeling, use_scaler=use_scaler, scaler=fitted_scaler,
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin_memory)

    weights = train_ds.get_class_weights()
    print(f"\n[Dataset] Class weights (train): not={weights[0]:.3f}, cheat={weights[1]:.3f}")
    print(f"[Dataset] Dim per batch: ({batch_size}, {seq_len}, {feature_dim})")

    return train_dl, valid_dl, test_dl


# Backward-compat alias
FEATURE_DIM = DEFAULT_FEATURE_DIM


# ─────────────────────────────────────────────────────────────────
# K-Fold Support
# ─────────────────────────────────────────────────────────────────
def pool_all_samples(feature_root, crop_root, labeling="any"):
    """Pool semua sampel dari train+valid+test untuk K-Fold CV."""
    all_samples = []
    for split in ["train", "valid", "test"]:
        feat_dir = Path(feature_root) / split
        crop_dir = Path(crop_root) / split
        if not feat_dir.exists():
            continue
        for video_dir in sorted(feat_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            crop_video = crop_dir / video_dir.name
            for npy in sorted(video_dir.glob("*.npy")):
                label = load_per_student_labels(crop_video, npy.stem, labeling)
                if label >= 0:
                    all_samples.append((npy, label))

    labels = [s[1] for s in all_samples]
    n_cheat = sum(labels)
    print(f"[KFold] Pooled {len(all_samples)} sampel: "
          f"cheat={n_cheat}, not_cheat={len(labels)-n_cheat}")
    return all_samples


class KFoldDataset(Dataset):
    """Dataset dari subset samples, untuk dipakai di K-Fold split."""
    def __init__(self, samples, seq_len=SEQ_LEN, feature_dim=DEFAULT_FEATURE_DIM,
                 scaler=None, augmentor=None):
        self.samples = samples
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.scaler = scaler
        self.augmentor = augmentor

    def __len__(self):
        return len(self.samples)

    def fit_scaler(self):
        from sklearn.preprocessing import StandardScaler
        data = np.concatenate([np.load(str(p)) for p, _ in self.samples], axis=0)
        self.scaler = StandardScaler().fit(data)
        return self.scaler

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        feat = np.load(str(npy_path)).astype(np.float32)
        if feat.shape[0] > self.seq_len:
            feat = feat[:self.seq_len]
        elif feat.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - feat.shape[0], self.feature_dim), dtype=np.float32)
            feat = np.concatenate([feat, pad], axis=0)
        if self.scaler is not None:
            feat = self.scaler.transform(feat).astype(np.float32)
        if self.augmentor is not None:
            feat = self.augmentor(feat)
        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)
