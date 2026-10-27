import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
from contextlib import redirect_stdout
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from benchmark.data import BenchmarkDataset, convert_labels
from benchmark.evaluation import predict_volume, score_predictions
from benchmark.objective import OriginalObjective
from benchmark.protocol import REFERENCE, read_ids, read_splits, validate_checkpoint
from benchmark.scar_augmentation import RandomGenerator
from benchmark.training import load_checkpoint, run_training, seed_everything, train_epoch
from utils import DiceLoss, ContrastiveLoss

torch.set_num_threads(1)


def reset(seed=23):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_official_split_matches_original():
    splits = read_splits(REFERENCE / "splits")
    assert len(splits["train"]) == 1972
    assert len(splits["test_vol"]) == 76
    root = Path(__file__).resolve().parents[1]
    for name, ids in splits.items():
        assert set(ids) == set(read_ids(root / "list" / f"{name}.txt"))


def test_reject_validation_and_modified_split(tmp_path):
    for name in ("train", "test_vol"):
        (tmp_path / f"{name}.txt").write_bytes((REFERENCE / "splits" / f"{name}.txt").read_bytes())
    (tmp_path / "val.txt").write_text("case0002")
    with pytest.raises(ValueError, match="validation"):
        read_splits(tmp_path)
    (tmp_path / "val.txt").unlink()
    train = tmp_path / "train.txt"
    train.write_text("\n".join(read_ids(train)[:-1]))
    with pytest.raises(ValueError, match="inventory"):
        read_splits(tmp_path)


@pytest.mark.parametrize("epoch", [0, 20, 21, 99, 100, 299])
def test_loss_and_gradients_equal_original_trainer(epoch):
    """Execute original trainer's loss block, rather than restating the formula in the test."""
    source = (REFERENCE / "original_trainer.py.txt").read_text()
    block = source[source.index("            ignores ="):source.index("            optimizer.zero_grad()")]
    import textwrap
    block = textwrap.dedent(block).replace("torch.zeros(1).cuda().float()", "torch.zeros(1).float()")
    reset()
    target = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]] * 8).unsqueeze(0)
    main = torch.randn(1, 4, 8, 8, requires_grad=True)
    dec = tuple(torch.randn_like(main, requires_grad=True) for _ in range(3))
    features = [[torch.randn(1, dim, size, size, requires_grad=True)
                 for size, dim in ((8, 3), (4, 4), (2, 5))] for _ in range(3)]
    text = [torch.randn(4, dim, requires_grad=True) for dim in (3, 4, 5)]
    # Use stripes that survive all three nearest-neighbour scales.
    target = torch.arange(8).remainder(4).repeat_interleave(2)[:8].repeat(8, 1).unsqueeze(0)
    target[:, :4, :4] = 0; target[:, :4, 4:] = 1
    target[:, 4:, :4] = 2; target[:, 4:, 4:] = 3
    args = SimpleNamespace(contrast_sample_num=10, contrast_w=0.1, region_fusion_start_epoch=100)
    env = dict(torch=torch, out_pre=main, dec_seg=dec, features_embedding_list=features,
               text_embedding_list=[t.repeat(2, 1) for t in text], label_batch=target,
               epoch_num=epoch, do_contrast=epoch > 20, args=args,
               con_loss=ContrastiveLoss(), ce_loss=nn.CrossEntropyLoss(), dice_loss=DiceLoss(4))
    reset(111)
    with redirect_stdout(io.StringIO()):
        exec(block, env)
    original = env["loss"]
    reset(111)
    with redirect_stdout(io.StringIO()):
        adapted = OriginalObjective()((main, dec, features, text), target, epoch)
    torch.testing.assert_close(original, adapted, rtol=0, atol=0)
    leaves = [main, *dec] + ([features[0][0], text[0]] if epoch > 20 else [])
    before = torch.autograd.grad(original.sum(), leaves, retain_graph=True)
    after = torch.autograd.grad(adapted.sum(), leaves)
    for x, y in zip(before, after):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_augmentation_identical_to_pinned_scar():
    source = (REFERENCE / "scar/training/dataset/myops_dataset.py").read_text(encoding="utf-8")
    import scipy.ndimage
    env = dict(np=np, torch=torch, random=random, ndimage=scipy.ndimage, F=torch.nn.functional)
    exec(source[source.index("def random_rot_flip("):source.index("class MyopsDataset(")], env)
    image = np.linspace(0, 1, 128 * 128, dtype=np.float32).reshape(128, 128)
    sample = dict(image=image, image1=image.copy(), image2=image.copy(), label=(image * 3).astype(np.int64))
    for seed in range(15):
        reset(seed)
        expected = env["RandomGenerator"]((128, 128))(sample)
        reset(seed)
        actual = RandomGenerator((128, 128))(sample)
        for key in sample:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def write_train_case(root, order, image_max=1):
    target = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    for modality in ("bSSFP", "LGE", "T2w"):
        folder = root / modality / "train_npz"
        folder.mkdir(parents=True)
        np.savez(folder / "case0002_slice000.npz", image=np.full((2, 2), image_max, np.float32),
                 label=target, label_order=order)


@pytest.mark.parametrize("order", ["legacy", "canonical"])
def test_reader_legacy_inside_no_double_normalization(tmp_path, order):
    lists = tmp_path / "lists"; lists.mkdir()
    (lists / "train.txt").write_text("case0002_slice000\n")
    write_train_case(tmp_path, order)
    sample = BenchmarkDataset(tmp_path, lists, "train", order)[0]
    np.testing.assert_array_equal(sample["label"], [[0, 1], [2, 3]] if order == "legacy" else [[0, 1], [3, 2]])
    assert sample["image"].max() == 1
    canonical = convert_labels(sample["label"], "legacy", "canonical")
    np.testing.assert_array_equal(canonical, [[0, 1], [3, 2]] if order == "legacy" else [[0, 1], [2, 3]])
    with pytest.raises(ValueError, match="conflicts"):
        BenchmarkDataset(tmp_path, lists, "train", "canonical" if order == "legacy" else "legacy")[0]


def test_reader_rejects_unnormalized_input(tmp_path):
    (tmp_path / "train.txt").write_text("case0002_slice000")
    write_train_case(tmp_path, "legacy", image_max=255)
    with pytest.raises(ValueError, match="normalization"):
        BenchmarkDataset(tmp_path, tmp_path, "train", "legacy")[0]


class MemoryTrain(Dataset):
    def __len__(self):
        return 1972

    def __getitem__(self, index):
        # Stochastic input catches failure to restore all augmentation RNGs.
        image = torch.ones(1, 2, 2) * (random.random() + np.random.rand() + index / 1972)
        return dict(image=image, image1=image, image2=image,
                    label=torch.tensor([[0, 1], [2, 3]], dtype=torch.int64))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x, x1, x2, do_contrast=False):
        logits = self.conv(self.drop(x + x1 + x2))
        return (logits, (logits, logits, logits), None, None) if self.training else logits


def train_config(output, lists, workers=0):
    return dict(device="cpu", amp="none", max_epochs=300, batch_size=300, accum_steps=2,
                num_workers=workers, seed=1234, base_lr=0.001, contrast_w=0.1, contrast_sample_num=10,
                start_contrast_epoch=20, region_fusion_start_epoch=100, output_dir=str(output),
                list_dir=str(lists), resume=None, epochs_per_run=2, pretrained=None)


@pytest.mark.parametrize("workers", [0, 1])
def test_resume_matches_uninterrupted_with_partial_accumulation(tmp_path, workers):
    sources = {"files_sha256": {"test": "locked"}}
    config = train_config(tmp_path / "full", REFERENCE / "splits", workers)
    seed_everything(1234)
    with redirect_stdout(io.StringIO()), pytest.warns(UserWarning, match="NOT equivalent"):
        full = run_training(TinyModel(), MemoryTrain(), config, {}, sources, {})
    config = train_config(tmp_path / "resumed", REFERENCE / "splits", workers)
    config["epochs_per_run"] = 1
    seed_everything(1234)
    with redirect_stdout(io.StringIO()), pytest.warns(UserWarning, match="NOT equivalent"):
        partial = run_training(TinyModel(), MemoryTrain(), config, {}, sources, {})
    checkpoint = load_checkpoint(partial)
    assert checkpoint["completed_epochs"] == 1
    assert not list(partial.parent.glob("best.*"))
    with pytest.raises(ValueError, match="Incomplete"):
        validate_checkpoint(checkpoint, require_complete=True)
    config["resume"] = str(partial)
    seed_everything(876)
    with redirect_stdout(io.StringIO()), pytest.warns(UserWarning, match="NOT equivalent"):
        resumed = run_training(TinyModel(), MemoryTrain(), config, {}, sources, {})
    full, resumed = load_checkpoint(full), load_checkpoint(resumed)
    assert full["global_step"] == resumed["global_step"] == 8
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key], resumed["model"][key], rtol=0, atol=0)
    assert full["scheduler"] == resumed["scheduler"]
    bad_config = dict(config, base_lr=0.002)
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="differs"):
        run_training(TinyModel(), MemoryTrain(), bad_config, {}, sources, {})


def test_inference_legacy_to_canonical_and_restore_mode():
    class ScarModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))
            self.calls = 0
        def forward(self, *images):
            assert not self.training
            self.calls += 1
            out = torch.zeros(images[0].shape[0], 4, 64, 64)
            out[:, 2] = 10  # legacy scar
            return out
    model = ScarModel()
    arrays = [np.zeros((128, 128, 5), dtype=np.float32)] * 3
    output = predict_volume(model, arrays, batch_size=2, device="cpu")
    assert output.shape == (128, 128, 5) and output.dtype == np.uint8
    assert np.all(output == 3) and model.calls == 3 and model.training


def test_shared_evaluator_76_synthetic_patients(tmp_path):
    ids = read_ids(REFERENCE / "splits/test_vol.txt")
    class TestVolumes:
        names = ids
        def __iter__(self):
            for case in ids:
                target = np.zeros((4, 4, 2), dtype=np.uint8)
                target[0, 0, 0] = 3
                yield dict(case_name=case, label=target)
    predictions = tmp_path / "predictions"; predictions.mkdir()
    for i, case in enumerate(ids):
        pred = np.zeros((4, 4, 2), dtype=np.uint8)
        if i != 0:
            pred[0, 0, 0] = 3
        np.savez_compressed(predictions / f"{case}_pred.npz", prediction=pred)
    report = score_predictions(predictions, TestVolumes(), tmp_path / "report", {})
    assert report["scar"]["mean_dice"] == 75 / 76
    assert report["scar"]["hd95_voxel_undefined_cases"] == 1
    assert report["edema"]["dice_undefined_cases"] == 76
    assert len((tmp_path / "report/per_case.csv").read_text().splitlines()) == 381
    np.savez(predictions / "extra_pred.npz", prediction=np.zeros((4, 4, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="76 locked"):
        score_predictions(predictions, TestVolumes(), tmp_path / "other", {})


def test_metric_source_byte_identical():
    from benchmark import scar_metrics
    assert Path(scar_metrics.__file__).read_bytes() == (REFERENCE / "scar/training/metrics/surface_distance.py").read_bytes()
