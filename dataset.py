"""
dataset.py v6 — Mendukung 2 Strategi Labeling Baru
====================================================
Strategi yang tersedia:

  1. "any"       (default lama, tidak direkomendasikan untuk imbalance)
                 Label = cheat jika ada minimal 1 frame cheat di sequence

  2. "majority"  (lebih konservatif)
                 Label = cheat jika >50% frame adalah cheat

  3. "threshold" (mid-ground, recommended kalau pakai per-student)
                 Label = cheat jika >X% frame adalah cheat (default 30%)

  4. "sliding"   (RECOMMENDED untuk dataset Fikar)
                 Pecah sequence siswa jadi banyak window kecil,
                 tiap window punya label sendiri (mayoritas dari window itu).
                 Datasetnya jadi jauh lebih besar.

Auto-detect feature_dim dari file .npy pertama.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Literal, List
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

DEFAULT_SEQ_LEN     = 8
DEFAULT_FEATURE_DIM = 38

SEQ_LEN     = DEFAULT_SEQ_LEN
FEATURE_DIM = DEFAULT_FEATURE_DIM


# ─────────────────────────────────────────────────────────────────
# Auto-detect feature_dim
# ─────────────────────────────────────────────────────────────────
def detect_feature_dim(feature_root: str) -> int:
    feature_path = Path(feature_root)
    for split in ["train", "valid", "test"]:
        split_dir = feature_path / split
        if not split_dir.exists():
            continue
        for video_dir in split_dir.iterdir():
            if not video_dir.is_dir():
                continue
            for npy_file in video_dir.glob("*.npy"):
                arr = np.load(str(npy_file))
                return int(arr.shape[1])
    return DEFAULT_FEATURE_DIM


# ─────────────────────────────────────────────────────────────────
# Label per-siswa (4 strategi)
# ─────────────────────────────────────────────────────────────────
def aggregate_labels(
    class_per_frame: List[int],
    strategy: str = "majority",
    threshold: float = 0.3,
) -> int:
    """
    Aggregasi label per-siswa dari list class per-frame.

    class_per_frame: list dari class YOLO (0=cheat, 1=not_cheat, -1=unmatched)
    Return: 0=not_cheat, 1=cheat, -1=tidak ada label

    PENTING: di file .npy hasil GRU ini, kita pakai konvensi:
      0 = not_cheat (kelas negatif)
      1 = cheat (kelas positif yang ingin dideteksi)
    Sehingga harus FLIP dari YOLO (di YOLO: 0=cheat, 1=not_cheat)
    """
    valid_labels = [c for c in class_per_frame if c >= 0]
    if not valid_labels:
        return -1

    n_cheat_frames    = sum(1 for c in valid_labels if c == 0)  # YOLO 0 = cheat
    n_total_frames    = len(valid_labels)
    cheat_ratio       = n_cheat_frames / n_total_frames

    if strategy == "any":
        # Cheat jika ada minimal 1 frame cheat
        return 1 if n_cheat_frames > 0 else 0
    elif strategy == "majority":
        # Cheat jika >50% frame adalah cheat
        return 1 if cheat_ratio > 0.5 else 0
    elif strategy == "threshold":
        # Cheat jika >X% frame adalah cheat
        return 1 if cheat_ratio > threshold else 0
    else:
        raise ValueError(f"Strategi tidak dikenal: {strategy}")


def load_per_student_labels(
    crop_video_dir: Path,
    student_id: str,
    strategy: str = "majority",
    threshold: float = 0.3,
) -> int:
    """Baca label per-siswa dari labels_per_student.json."""
    labels_json_path = crop_video_dir / "labels_per_student.json"
    if not labels_json_path.exists():
        return -1

    with open(labels_json_path, "r") as f:
        all_labels = json.load(f)

    student_classes = all_labels.get(student_id, [])
    if not student_classes:
        return -1

    return aggregate_labels(student_classes, strategy, threshold)


# ─────────────────────────────────────────────────────────────────
# OPSI A: Dataset Per-Siswa (1 student = 1 sequence = 1 label)
# ─────────────────────────────────────────────────────────────────
class ExamCheatingDataset(Dataset):
    """
    Pendekatan klasik: 1 siswa = 1 sample.

    Penggunaan:
        ds = ExamCheatingDataset(
            feature_root='features',
            crop_root='crop',
            split='train',
            labeling='majority',      # ← rekomendasi default
            threshold=0.3,            # untuk strategy='threshold'
        )
    """

    def __init__(
        self,
        feature_root: str,
        crop_root: str,
        split: str = "train",
        seq_len: int = DEFAULT_SEQ_LEN,
        feature_dim: Optional[int] = None,
        labeling: Literal["any", "majority", "threshold"] = "majority",
        threshold: float = 0.3,
        use_scaler: bool = False,
        scaler=None,
        augment: bool = False,
        verbose: bool = True,
    ):
        self.seq_len    = seq_len
        self.split      = split
        self.use_scaler = use_scaler
        self.augment    = augment and (split == "train")
        self.labeling   = labeling
        self.threshold  = threshold

        if feature_dim is None:
            feature_dim = detect_feature_dim(feature_root)
            if verbose:
                print(f"[Dataset/{split}] auto-detected feature_dim={feature_dim}")
        self.feature_dim = feature_dim

        feature_split_dir = Path(feature_root) / split
        crop_split_dir    = Path(crop_root)    / split

        self.samples = []
        for video_dir in sorted(feature_split_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            crop_video_dir = crop_split_dir / video_dir.name
            for npy_file in sorted(video_dir.glob("*.npy")):
                student_id = npy_file.stem
                label = load_per_student_labels(
                    crop_video_dir, student_id, labeling, threshold
                )
                self.samples.append((npy_file, label))

        valid_samples = [s for s in self.samples if s[1] >= 0]
        n_dropped = len(self.samples) - len(valid_samples)
        self.samples = valid_samples

        if verbose:
            dist = Counter([s[1] for s in self.samples])
            print(f"[Dataset/{split}] {len(self.samples)} sampel  "
                  f"({n_dropped} di-drop)  | "
                  f"dist: not_cheat(0)={dist.get(0,0)}, cheat(1)={dist.get(1,0)}  "
                  f"| strategy={labeling}")

        self.scaler = None
        if use_scaler:
            if scaler is not None:
                self.scaler = scaler
            elif split == "train":
                self.scaler = self._fit_scaler()
            else:
                raise ValueError("use_scaler=True tapi split bukan train.")

    def _fit_scaler(self):
        from sklearn.preprocessing import StandardScaler
        all_data = [np.load(str(p)) for p, _ in self.samples]
        all_data = np.concatenate(all_data, axis=0)
        scaler = StandardScaler()
        scaler.fit(all_data)
        return scaler

    def _augment(self, feat):
        feat = feat + np.random.randn(*feat.shape).astype(np.float32) * 0.02
        if self.seq_len >= 4 and np.random.rand() < 0.3:
            feat[np.random.randint(0, self.seq_len)] = 0.0
        return feat

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
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

        if self.augment:
            feat = self._augment(feat)

        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)

    def get_class_weights(self):
        labels = [s[1] for s in self.samples]
        n_total = len(labels)
        n_cheat = sum(labels)
        n_not   = n_total - n_cheat
        if n_cheat == 0 or n_not == 0:
            return torch.ones(2)
        return torch.tensor([n_total / (2 * n_not),
                             n_total / (2 * n_cheat)], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────
# OPSI B: Dataset Sliding Window (RECOMMENDED)
# ─────────────────────────────────────────────────────────────────
class ExamCheatingDatasetSliding(Dataset):
    """
    Sliding window dataset: setiap siswa dipecah jadi banyak window kecil.

    Contoh: siswa A punya 80 frame, window_size=8, stride=4
            → menghasilkan ~18 window per siswa
            Setiap window dilabeli berdasarkan mayoritas frame di window itu.

    PENTING: TIDAK ADA LEAKAGE karena:
      - Train/valid/test sudah dipisah di level VIDEO sebelum windowing
      - Window dari video valid TIDAK PERNAH masuk ke train

    Args:
        window_size : panjang window (= seq_len model)
        stride      : jarak antar window (kecil = banyak overlap = banyak data)
        min_valid_ratio : minimum frame valid dalam window agar dipakai
                          (e.g. 0.5 = minimal 50% frame harus punya label)
    """

    def __init__(
        self,
        feature_root: str,
        crop_root: str,
        split: str = "train",
        window_size: int = 8,
        stride: int = 4,
        feature_dim: Optional[int] = None,
        labeling: Literal["majority", "threshold"] = "majority",
        threshold: float = 0.3,
        min_valid_ratio: float = 0.5,
        use_scaler: bool = False,
        scaler=None,
        augment: bool = False,
        verbose: bool = True,
    ):
        self.window_size = window_size
        self.stride      = stride
        self.split       = split
        self.use_scaler  = use_scaler
        self.augment     = augment and (split == "train")

        if feature_dim is None:
            feature_dim = detect_feature_dim(feature_root)
            if verbose:
                print(f"[SlidingDataset/{split}] auto-detected feature_dim={feature_dim}")
        self.feature_dim = feature_dim

        feature_split_dir = Path(feature_root) / split
        crop_split_dir    = Path(crop_root)    / split

        # self.windows: list of (feat_array, label)
        # PRELOAD karena windowing butuh akses raw .npy + class per-frame
        self.windows = []
        n_students_processed = 0

        for video_dir in sorted(feature_split_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            crop_video_dir = crop_split_dir / video_dir.name
            labels_json    = crop_video_dir / "labels_per_student.json"
            if not labels_json.exists():
                if verbose:
                    print(f"  Skip {video_dir.name}: labels_per_student.json tidak ada")
                continue

            with open(labels_json) as f:
                all_labels = json.load(f)

            for npy_file in sorted(video_dir.glob("*.npy")):
                student_id      = npy_file.stem
                student_classes = all_labels.get(student_id, [])

                if not student_classes:
                    continue

                feat_array = np.load(str(npy_file))  # (T, D)
                T = feat_array.shape[0]

                # JUMLAH FRAME efektif = min(jumlah class_per_frame, T)
                # karena bisa beda (kalau sequence di-pad/truncate di Fase 1)
                T_eff = min(T, len(student_classes))

                # Generate window dengan sliding
                for start in range(0, T_eff - window_size + 1, stride):
                    end = start + window_size

                    # Class per-frame untuk window ini
                    window_classes = student_classes[start:end]
                    valid_count = sum(1 for c in window_classes if c >= 0)
                    if valid_count / window_size < min_valid_ratio:
                        continue  # window kurang reliable

                    # Aggregasi → label window
                    label = aggregate_labels(window_classes, labeling, threshold)
                    if label < 0:
                        continue

                    # Ambil fitur window
                    window_feat = feat_array[start:end]
                    self.windows.append((window_feat.copy(), label))

                n_students_processed += 1

        if verbose:
            dist = Counter([w[1] for w in self.windows])
            print(f"[SlidingDataset/{split}] "
                  f"{n_students_processed} siswa → {len(self.windows)} windows  | "
                  f"dist: not_cheat(0)={dist.get(0,0)}, cheat(1)={dist.get(1,0)}  "
                  f"| window={window_size} stride={stride} strategy={labeling}")

        # Scaler
        self.scaler = None
        if use_scaler:
            if scaler is not None:
                self.scaler = scaler
            elif split == "train":
                self.scaler = self._fit_scaler()
            else:
                raise ValueError("use_scaler=True tapi split bukan train.")

    def _fit_scaler(self):
        from sklearn.preprocessing import StandardScaler
        all_data = np.concatenate([w[0] for w in self.windows], axis=0)
        scaler = StandardScaler()
        scaler.fit(all_data)
        return scaler

    def _augment(self, feat):
        feat = feat + np.random.randn(*feat.shape).astype(np.float32) * 0.02
        if self.window_size >= 4 and np.random.rand() < 0.3:
            feat[np.random.randint(0, self.window_size)] = 0.0
        return feat

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        feat, label = self.windows[idx]
        feat = feat.astype(np.float32)

        if self.scaler is not None:
            feat = self.scaler.transform(feat).astype(np.float32)

        if self.augment:
            feat = self._augment(feat)

        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)

    def get_class_weights(self):
        labels = [w[1] for w in self.windows]
        n_total = len(labels)
        n_cheat = sum(labels)
        n_not   = n_total - n_cheat
        if n_cheat == 0 or n_not == 0:
            return torch.ones(2)
        return torch.tensor([n_total / (2 * n_not),
                             n_total / (2 * n_cheat)], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────
def build_dataloaders(
    feature_root: str,
    crop_root: str,
    seq_len: int = DEFAULT_SEQ_LEN,
    feature_dim: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    use_scaler: bool = False,
    labeling: str = "majority",         # ← default diubah dari "any"
    threshold: float = 0.3,
    pin_memory: bool = True,
    augment: bool = False,
    # Sliding window options
    use_sliding: bool = False,
    stride: int = 4,
    min_valid_ratio: float = 0.5,
):
    """
    Build train/valid/test DataLoaders.

    Set use_sliding=True untuk pakai sliding window (RECOMMENDED).
    """
    common_kwargs = dict(
        feature_root=feature_root, crop_root=crop_root,
        feature_dim=feature_dim, labeling=labeling,
        threshold=threshold, use_scaler=use_scaler,
    )

    if use_sliding:
        DatasetClass = ExamCheatingDatasetSliding
        common_kwargs["window_size"]     = seq_len
        common_kwargs["stride"]          = stride
        common_kwargs["min_valid_ratio"] = min_valid_ratio
    else:
        DatasetClass = ExamCheatingDataset
        common_kwargs["seq_len"] = seq_len

    train_ds = DatasetClass(split="train", augment=augment, **common_kwargs)
    fitted_scaler = train_ds.scaler if use_scaler else None
    actual_dim    = train_ds.feature_dim

    common_kwargs["scaler"]      = fitted_scaler
    common_kwargs["feature_dim"] = actual_dim

    valid_ds = DatasetClass(split="valid", **common_kwargs)
    test_ds  = DatasetClass(split="test",  **common_kwargs)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory,
                          drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)

    weights = train_ds.get_class_weights()
    print(f"\n[Dataset] feat_dim={actual_dim} | "
          f"weights: not_cheat={weights[0]:.3f}, cheat={weights[1]:.3f}")

    return train_dl, valid_dl, test_dl
