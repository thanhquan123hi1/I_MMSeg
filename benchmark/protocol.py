"""Data, source and fixed-budget contracts. No model evaluation occurs here."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path(__file__).resolve().parent / "reference"
PROTOCOL = "myops380_train304_test76_last_v1"
TRAIN_COUNT = 1972
EPOCHS = 300
MODALITIES = ("bSSFP", "LGE", "T2w")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_ids(path):
    names = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"Empty or duplicate IDs: {path}")
    return names


def ids_hash(names):
    return hashlib.sha256(("\n".join(sorted(names)) + "\n").encode()).hexdigest()


def read_splits(list_dir):
    list_dir = Path(list_dir)
    if any((list_dir / f"{name}.txt").exists() for name in ("val", "val_vol")):
        raise ValueError("Use a no-validation manifest directory containing only train/test manifests")
    splits = {name: read_ids(list_dir / f"{name}.txt") for name in ("train", "test_vol")}
    for name, ids in splits.items():
        reference = read_ids(REFERENCE / "splits" / f"{name}.txt")
        expected_hash = {"train": "16248ec3630bb0493fb421aab5ffb4dc9b1d0fd49c8614d69d3098a92305f919",
                         "test_vol": "3237b31411833cedcb3ecd86d36edae356bd3f880d3fcfd915066f1c850d9cbc"}
        # Official reference hashes are checked below; never trust a mutable list as the only reference.
        if ids_hash(reference) != expected_hash[name]:
            raise ValueError("Pinned reference changed")
        if set(ids) != set(reference):
            raise ValueError(f"{name} differs from the pinned official patient/slice inventory")
    patients = {re.fullmatch(r"(case\d{4})_slice\d{3}", x)[1] for x in splits["train"]}
    if (len(splits["train"]) != TRAIN_COUNT or len(patients) != 304
            or len(splits["test_vol"]) != 76 or patients & set(splits["test_vol"])):
        raise ValueError("Requires 1972 slices, 304 train patients and 76 disjoint test patients")
    return splits


def lock_data(data_root, list_dir, label_order):
    splits = read_splits(list_dir)
    root = Path(data_root)
    digest = hashlib.sha256()
    files = []
    for modality in MODALITIES:
        for split, folder, suffix in (("train", "train_npz", ".npz"),
                                      ("test_vol", "test_vol_h5", ".npy.h5")):
            paths = list((root / modality / folder).glob("*" + suffix))
            if {p.name[:-len(suffix)] for p in paths} != set(splits[split]):
                raise ValueError(f"Cache inventory mismatch: {modality}/{folder}")
            files.extend(paths)
    metadata = root / "dataset_metadata.json"
    if metadata.is_file():
        files.append(metadata)
    # Deliberately excludes unused legacy validation volumes, matching SCAR no-val.
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        digest.update(f"{path.relative_to(root).as_posix()}\0{sha256(path)}\n".encode())
    return {"protocol": PROTOCOL, "cache_sha256": digest.hexdigest(),
            "cache_files": len(files), "source_label_order": label_order,
            "split_ids_sha256": {k: ids_hash(v) for k, v in splits.items()},
            "manifest_sha256": {k: sha256(Path(list_dir) / f"{k}.txt") for k in splits},
            "train_patients": 304, "train_slices": TRAIN_COUNT, "test_patients": 76}


def source_record():
    manifest = json.loads((REFERENCE / "manifest.json").read_text())
    for name, expected in manifest["source_files_sha256"].items():
        if sha256(REFERENCE / "scar" / name) != expected:
            raise ValueError(f"Pinned SCAR source changed: {name}")
    if sha256(ROOT / "benchmark/scar_metrics.py") != manifest["source_files_sha256"]["training/metrics/surface_distance.py"]:
        raise ValueError("Metric implementation differs from the pinned SCAR source")
    paths = [ROOT / p for p in ("train.py", "test.py", "evaluate.py", "evaluate_predictions.py", "utils.py", "trainer.py")]
    paths += list((ROOT / "networks").glob("*.py"))
    paths += [p for p in (ROOT / "benchmark").rglob("*")
              if p.is_file() and p.suffix in {".py", ".json", ".txt", ".yaml", ".md"}]
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    try:
        commit, dirty = git("rev-parse", "HEAD"), git("status", "--short")
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, "Git unavailable; content hashes retained"
    return {"files_sha256": hashes, "commit": commit, "working_tree": dirty,
            "scar_snapshot": manifest}


def validate_checkpoint(checkpoint, require_complete=False):
    required = {"config", "completed_epochs", "planned_epochs", "total_updates", "global_step",
                "amp_skipped_updates", "scheduler", "checkpoint_selection_metric"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Expected a full no-val training checkpoint, not legacy state_dict weights")
    config = checkpoint["config"]
    if not isinstance(config, dict):
        raise ValueError("Invalid checkpoint config")
    integers = [checkpoint.get(key) for key in ("epoch", "completed_epochs", "planned_epochs", "total_updates",
                                               "global_step", "amp_skipped_updates")]
    integers += [config.get(key) for key in ("max_epochs", "batch_size", "accum_steps")]
    if any(type(value) is not int for value in integers):
        raise ValueError("Checkpoint epoch/update/batch counters must be integers")
    completed, planned = checkpoint["completed_epochs"], checkpoint["planned_epochs"]
    if (checkpoint.get("protocol") != PROTOCOL or planned != EPOCHS
            or config["max_epochs"] != EPOCHS or not 0 < completed <= planned
            or checkpoint.get("epoch") != completed - 1
            or checkpoint.get("checkpoint_selection") != "last_completed_epoch"
            or checkpoint.get("checkpoint_selection_metric") is not None
            or checkpoint.get("training_complete") is not (completed == planned)):
        raise ValueError("Invalid final-checkpoint protocol/budget metadata")
    batch, accum = config["batch_size"], config["accum_steps"]
    if batch < 1 or accum < 1:
        raise ValueError("Invalid batch/accumulation")
    per_epoch = ((TRAIN_COUNT + batch - 1) // batch + accum - 1) // accum
    if (checkpoint["total_updates"] != planned * per_epoch
            or checkpoint["global_step"] < 1 or checkpoint["amp_skipped_updates"] < 0
            or checkpoint["global_step"] + checkpoint["amp_skipped_updates"] != completed * per_epoch
            or checkpoint["scheduler"]["last_epoch"] != checkpoint["global_step"]):
        raise ValueError("Inconsistent update/scheduler budget")
    if require_complete and completed != EPOCHS:
        raise ValueError(f"Incomplete training: {completed}/{EPOCHS}; resume before final evaluation")
    return checkpoint
