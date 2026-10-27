"""Aligned cache reader. Train targets are legacy; evaluation targets canonical."""
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset

from .protocol import MODALITIES, read_ids
from .scar_augmentation import RandomGenerator

ORDERS = {"legacy": "legacy", "background_normal_scar_edema": "legacy",
          "canonical": "canonical", "background_normal_edema_scar": "canonical"}
SWAP = np.asarray([0, 1, 3, 2], dtype=np.uint8)
METADATA_KEYS = ("label_order", "axis_order", "patient_id", "spacing", "affine", "spacing_unit", "normalization")


def scalar_metadata(metadata, key, default=None):
    if key not in metadata:
        return default
    value = np.asarray(metadata[key])
    if value.size != 1:
        raise ValueError(f"{key} metadata must be scalar")
    value = value.item()
    return value.decode() if isinstance(value, bytes) else value


def check_metadata(metadata, name, ndim):
    """Reject mismatched identities/physical grids even if array shapes happen to match."""
    patient = name.split("_slice", 1)[0]
    for item in metadata:
        if scalar_metadata(item, "patient_id", patient) != patient:
            raise ValueError(f"patient_id metadata disagrees with filename: {name}")
        axis = scalar_metadata(item, "axis_order", "HW" if ndim == 2 else "HWD")
        if axis != ("HW" if ndim == 2 else "HWD"):
            raise ValueError(f"Unsupported axis order for {name}")
        if "spacing" in item or "affine" in item:
            spacing = np.asarray(item.get("spacing", []), dtype=float)
            affine = np.asarray(item.get("affine", []), dtype=float)
            if (spacing.shape != (3,) or affine.shape != (4, 4) or not np.isfinite(spacing).all()
                    or not np.isfinite(affine).all() or (spacing <= 0).any()
                    or abs(np.linalg.det(affine[:3, :3])) <= 1e-12
                    or not np.allclose(np.linalg.norm(affine[:3, :3], axis=0), spacing, atol=1e-4, rtol=1e-4)):
                raise ValueError(f"Invalid spacing/affine geometry for {name}")
        if scalar_metadata(item, "spacing_unit", "unknown") not in {"mm", "unknown"}:
            raise ValueError(f"Unsupported spacing unit for {name}")
    first = metadata[0]
    for item in metadata[1:]:
        for key in ("spacing", "affine", "spacing_unit", "normalization"):
            if (key in first) != (key in item):
                raise ValueError(f"Inconsistent {key} metadata between modalities: {name}")
            if key in first:
                same = (np.allclose(first[key], item[key], atol=1e-4, rtol=1e-4)
                        if key in {"spacing", "affine"}
                        else scalar_metadata(first, key) == scalar_metadata(item, key))
                if not same:
                    raise ValueError(f"Incompatible {key} between modalities: {name}")


def convert_labels(label, source, destination):
    label = np.asarray(label)
    if not np.isfinite(label).all() or not np.isin(label, [0, 1, 2, 3]).all():
        raise ValueError("Labels must be integers 0..3")
    source, destination = ORDERS[source], ORDERS[destination]
    return label.astype(np.uint8) if source == destination else SWAP[label.astype(np.uint8)]


class BenchmarkDataset(Dataset):
    def __init__(self, data_root, list_dir, split, label_order, augment=False):
        if split not in {"train", "test_vol"} or (augment and split != "train"):
            raise ValueError("Only train/test_vol are supported; augmentation is train-only")
        self.root, self.split = Path(data_root), split
        self.order = ORDERS[label_order]
        self.names = read_ids(Path(list_dir) / f"{split}.txt")
        self.transform = RandomGenerator((128, 128)) if augment else None

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        if self.split == "train":
            folder, suffix, ndim, destination = "train_npz", ".npz", 2, "legacy"
        else:
            folder, suffix, ndim, destination = "test_vol_h5", ".npy.h5", 3, "canonical"
        images, labels, metadata = [], [], []
        for modality in MODALITIES:
            path = self.root / modality / folder / (name + suffix)
            if suffix == ".npz":
                with np.load(path, allow_pickle=False) as data:
                    image, label = data["image"].copy(), data["label"].copy()
                    info = {key: data[key].copy() for key in METADATA_KEYS if key in data}
            else:
                with h5py.File(path, "r") as data:
                    image, label = data["image"][:], data["label"][:]
                    info = dict(data.attrs)
                    for key in ("spacing", "affine"):
                        if key in data:
                            info[key] = data[key][:]
            stored_order = scalar_metadata(info, "label_order")
            if stored_order is not None and ORDERS[stored_order] != self.order:
                raise ValueError(f"label_order conflicts with cache metadata: {path}")
            if (image.ndim != ndim or image.shape != label.shape or not image.size
                    or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1):
                raise ValueError(f"Expected aligned finite [0,1] cache, no normalization repeated: {path}")
            images.append(image.astype(np.float32))
            labels.append(convert_labels(label, self.order, destination))
            metadata.append(info)
        if any(x.shape != images[0].shape for x in images) or any(not np.array_equal(x, labels[0]) for x in labels):
            raise ValueError(f"Modalities/masks differ for {name}")
        check_metadata(metadata, name, ndim)
        sample = dict(zip(("image", "image1", "image2"), images))
        sample.update(label=labels[0].astype(np.int64), case_name=name)
        if self.transform:
            sample = self.transform(sample)
        return sample
