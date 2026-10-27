"""Regression tests for defects found during the second benchmark audit."""
import copy
import json

import h5py
import numpy as np
import pytest
import torch

from benchmark.data import BenchmarkDataset
from benchmark.evaluation import score_predictions, predict_volume
from benchmark.protocol import REFERENCE, read_ids, validate_checkpoint
from benchmark.training import float_outputs, train_epoch
from test_protocol_guards import completed_checkpoint


def write_cache(root, split="train", metadata=None):
    name = "case0002_slice000" if split == "train" else "case0001"
    (root / f"{split}.txt").write_text(name)
    shape = (4, 4) if split == "train" else (4, 4, 2)
    metadata = metadata or {}
    for modality in ("bSSFP", "LGE", "T2w"):
        folder = root / modality / ("train_npz" if split == "train" else "test_vol_h5")
        folder.mkdir(parents=True)
        extra = metadata.get(modality, {})
        if split == "train":
            np.savez(folder / f"{name}.npz", image=np.ones(shape, np.float32),
                     label=np.ones(shape, np.uint8), **extra)
        else:
            with h5py.File(folder / f"{name}.npy.h5", "w") as stream:
                stream["image"] = np.ones(shape, np.float32)
                stream["label"] = np.ones(shape, np.uint8)
                for key, value in extra.items():
                    if key in {"spacing", "affine"}:
                        stream[key] = value
                    else:
                        stream.attrs[key] = value


@pytest.mark.parametrize("split", ["train", "test_vol"])
def test_reader_rejects_wrong_patient_identity(tmp_path, split):
    write_cache(tmp_path, split, {"LGE": {"patient_id": "case0999"}})
    with pytest.raises(ValueError, match="patient_id"):
        BenchmarkDataset(tmp_path, tmp_path, split, "legacy")[0]


@pytest.mark.parametrize("split", ["train", "test_vol"])
def test_reader_rejects_equal_shape_but_different_geometry(tmp_path, split):
    shifted = np.eye(4); shifted[0, 3] = 10
    meta = {modality: dict(spacing=np.ones(3), affine=np.eye(4), spacing_unit="mm")
            for modality in ("bSSFP", "LGE", "T2w")}
    meta["LGE"]["affine"] = shifted
    write_cache(tmp_path, split, meta)
    with pytest.raises(ValueError, match="affine"):
        BenchmarkDataset(tmp_path, tmp_path, split, "legacy")[0]


def test_metadata_cannot_override_computed_scores(tmp_path):
    ids = read_ids(REFERENCE / "splits/test_vol.txt")
    class Volumes:
        names = ids
        def __iter__(self):
            for case in ids:
                label = np.zeros((2, 2, 2), np.uint8); label[0, 0, 0] = 3
                yield dict(case_name=case, label=label)
    pred = tmp_path / "predictions"; pred.mkdir()
    for case in ids:
        np.savez(pred / f"{case}_pred.npz", prediction=np.zeros((2, 2, 2), np.uint8))
    report = score_predictions(pred, Volumes(), tmp_path / "report", {"scar": {"mean_dice": 1.0}})
    assert report["scar"]["mean_dice"] == 0.0


@pytest.mark.parametrize("location,key,value", [("root", "global_step", 37200.0),
        ("root", "amp_skipped_updates", False), ("config", "batch_size", 16.0)])
def test_checkpoint_rejects_noninteger_counters(location, key, value):
    checkpoint = completed_checkpoint()
    target = checkpoint if location == "root" else checkpoint["config"]
    target[key] = value
    with pytest.raises(ValueError, match="integers"):
        validate_checkpoint(checkpoint, require_complete=True)


def test_fp16_is_rejected_before_inference():
    model = torch.nn.Conv2d(1, 4, 1)
    arrays = [np.ones((2, 2, 1), np.float32)] * 3
    with pytest.raises(ValueError, match="FP16"):
        predict_volume(model, arrays, device="cpu", amp_dtype=torch.float16)


def test_loss_precision_adapter_preserves_autograd():
    value = torch.tensor([2.], dtype=torch.bfloat16, requires_grad=True)
    converted = float_outputs((value, [value], None))
    assert converted[0].dtype == torch.float32 and converted[2] is None
    converted[0].square().sum().backward()
    assert value.grad.item() == 4


def test_full_300_epoch_control_flow_with_tiny_synthetic_model(monkeypatch, tmp_path):
    """Exercise all 300 epochs/checkpoints; objective equivalence is tested separately."""
    import io
    from contextlib import redirect_stdout
    from benchmark import training
    class TinyData(torch.utils.data.Dataset):
        def __len__(self):
            return 1972
        def __getitem__(self, index):
            image = torch.ones(1, 1, 1)
            return dict(image=image, image1=image, image2=image, label=torch.zeros(1, 1, dtype=torch.long))
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.conv = torch.nn.Conv2d(1, 4, 1)
        def forward(self, x, x1, x2, do_contrast):
            return self.conv(x)
    class ControlFlowLoss(torch.nn.Module):
        def __init__(self, *args):
            super().__init__()
        def do_contrast(self, epoch):
            return False
        def forward(self, outputs, target, epoch):
            return torch.nn.functional.cross_entropy(outputs, target)
    monkeypatch.setattr(training, "OriginalObjective", ControlFlowLoss)
    config = dict(device="cpu", amp="none", max_epochs=300, batch_size=1972, accum_steps=1,
                  num_workers=0, seed=1234, base_lr=0.001, contrast_w=0.1, contrast_sample_num=10,
                  start_contrast_epoch=20, region_fusion_start_epoch=100, output_dir=str(tmp_path / "run"),
                  list_dir=str(REFERENCE / "splits"), resume=None, epochs_per_run=0, pretrained=None)
    with redirect_stdout(io.StringIO()):
        path = training.run_training(TinyModel(), TinyData(), config, {}, {"files_sha256": {}}, {})
    checkpoint = training.load_checkpoint(path)
    validate_checkpoint(checkpoint, require_complete=True)
    assert checkpoint["completed_epochs"] == checkpoint["global_step"] == 300
    assert checkpoint["scheduler"]["_last_lr"] == [0.0]
    assert len((path.parent / "train.jsonl").read_text().splitlines()) == 300
    assert not list(path.parent.glob("best.*"))
