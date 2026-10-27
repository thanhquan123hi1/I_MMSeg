import copy
import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from benchmark.model import load_encoder_weights
from benchmark.protocol import REFERENCE, PROTOCOL, validate_checkpoint


def completed_checkpoint():
    return dict(config=dict(max_epochs=300, batch_size=16, accum_steps=1),
                protocol=PROTOCOL, epoch=299, completed_epochs=300, planned_epochs=300,
                checkpoint_selection="last_completed_epoch", checkpoint_selection_metric=None,
                training_complete=True, total_updates=37200, global_step=37200,
                amp_skipped_updates=0, scheduler={"last_epoch": 37200})


def test_final_budget_and_tamper_guards():
    checkpoint = completed_checkpoint()
    validate_checkpoint(checkpoint, require_complete=True)
    for key, bad in (("epoch", 298), ("training_complete", False), ("total_updates", 300),
                     ("global_step", 37199), ("checkpoint_selection_metric", "test_dice")):
        with pytest.raises(ValueError):
            validate_checkpoint(dict(checkpoint, **{key: bad}), require_complete=True)
    with pytest.raises(ValueError, match="legacy"):
        validate_checkpoint({"conv.weight": torch.ones(1)}, require_complete=True)


def test_cli_train_constructs_only_train_dataset(monkeypatch, tmp_path):
    import train
    seen = []
    monkeypatch.setattr(train, "lock_data", lambda *a: {})
    monkeypatch.setattr(train, "source_record", lambda: {})
    monkeypatch.setattr(train, "initialization_record", lambda c: {})
    def dataset(*args, **kwargs):
        seen.append(args[2])
        return object()
    monkeypatch.setattr(train, "BenchmarkDataset", dataset)
    monkeypatch.setattr(train, "build_model", lambda *a, **kw: object())
    monkeypatch.setattr(train, "run_training", lambda *a: "synthetic-no-execution")
    train.main(["--data-root", str(tmp_path), "--label-order", "legacy", "--output-dir", str(tmp_path / "out")])
    assert seen == ["train"]


def test_pretrained_mapping_checks_all_encoders_before_mutation(tmp_path):
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.root = nn.Module()
            self.root.conv = nn.Conv2d(3, 2, 1, bias=False)
            self.root.gn = nn.GroupNorm(1, 2)
            self.body = nn.ModuleDict()
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            for i in (1, 2, 3):
                encoder = nn.Module(); encoder.embeddings = nn.Module()
                encoder.embeddings.hybrid_model = Backbone()
                setattr(self, f"transformer{i}", encoder)
    model = Model()
    path = tmp_path / "encoder.npz"
    np.savez(path, **{"conv_root/kernel": np.full((1, 1, 3, 2), 7, np.float32),
                     "gn_root/scale": np.full(2, 3, np.float32),
                     "gn_root/bias": np.full(2, 5, np.float32)})
    assert load_encoder_weights(model, path) == 9
    for i in (1, 2, 3):
        root = getattr(model, f"transformer{i}").embeddings.hybrid_model.root
        assert torch.all(root.conv.weight == 7) and torch.all(root.gn.weight == 3) and torch.all(root.gn.bias == 5)
    before = copy.deepcopy(model.state_dict())
    np.savez(path, **{"conv_root/kernel": np.full((1, 1, 3, 2), 8, np.float32),
                     "gn_root/scale": np.full(20, 3, np.float32), "gn_root/bias": np.full(2, 5, np.float32)})
    with pytest.raises(ValueError):
        load_encoder_weights(model, path)
    for key, value in before.items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


def test_shared_cli_accepts_completed_scar_export_only(monkeypatch, tmp_path):
    import json
    import evaluate_predictions as cli
    from benchmark.protocol import read_ids, sha256
    ids = read_ids(REFERENCE / "splits/test_vol.txt")
    class Volumes:
        names = ids
        def __iter__(self):
            for case in ids:
                yield dict(case_name=case, label=np.zeros((2, 2, 2), dtype=np.uint8))
    lock = {"cache_sha256": "synthetic", "split_ids_sha256": {"train": "fixed", "test_vol": "fixed"},
            "source_label_order": "background_normal_scar_edema"}
    monkeypatch.setattr(cli, "lock_data", lambda *a: lock)
    monkeypatch.setattr(cli, "BenchmarkDataset", lambda *a: Volumes())
    path = tmp_path / "last.pt"
    checkpoint = completed_checkpoint()
    checkpoint.update(args=dict(no_val=True, max_epochs=300, batch_size=16, accum_steps=1,
                                data_root="synthetic", list_dir="synthetic", label_order="legacy"),
                      training_protocol={"id": PROTOCOL}, benchmark_data=lock,
                      model_config={"ablation": "M2-Plus"})
    torch.save(checkpoint, path)
    predictions = tmp_path / "predictions"; predictions.mkdir()
    for case in ids:
        np.savez(predictions / f"{case}_pred.npz", prediction=np.zeros((2, 2, 2), dtype=np.uint8))
    (predictions / "metrics.json").write_text(json.dumps(dict(checkpoint_sha256=sha256(path),
                                            inference_protocol={"id": "single_pass_argmax_v1"})))
    cli.main(["--model", "M2-Plus", "--checkpoint", str(path), "--predictions", str(predictions),
              "--output-dir", str(tmp_path / "report")])
    report = json.loads((tmp_path / "report/metrics.json").read_text())
    assert report["case_count"] == 76 and report["model"] == "M2-Plus"
    with pytest.raises(ValueError, match="label order"):
        cli.main(["--model", "M2-Plus", "--checkpoint", str(path), "--predictions", str(predictions),
                  "--label-order", "canonical", "--output-dir", str(tmp_path / "wrong_labels")])
    checkpoint["completed_epochs"] = 299
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="completed"):
        cli.main(["--model", "M2-Plus", "--checkpoint", str(path), "--predictions", str(predictions),
                  "--output-dir", str(tmp_path / "reject")])
