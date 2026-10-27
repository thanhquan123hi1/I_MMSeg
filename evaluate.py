"""Export and score only a completed 300-epoch I-MMSeg last.pth."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark.data import BenchmarkDataset
from benchmark.evaluation import INFERENCE, predict_volume, score_predictions
from benchmark.model import build_model, initialization_record
from benchmark.protocol import lock_data, sha256, source_record, validate_checkpoint
from benchmark.training import load_checkpoint


def validate_run(path, data_root=None, list_dir=None, label_order=None):
    path = Path(path)
    if path.name != "last.pth":
        raise ValueError("Only the completed last.pth is eligible")
    checkpoint = validate_checkpoint(load_checkpoint(path), require_complete=True)
    config = checkpoint["config"]
    data_root = data_root or config["data_root"]
    list_dir = list_dir or config["list_dir"]
    label_order = label_order or config["label_order"]
    if lock_data(data_root, list_dir, label_order) != checkpoint["data_lock"]:
        raise ValueError("Evaluation cache/split/order differs from training")
    if source_record()["files_sha256"] != checkpoint["sources"]["files_sha256"]:
        raise ValueError("Source/reference snapshot differs from training")
    text = initialization_record({"pretrained": None})["text_embedding_sha256"]
    if text != checkpoint["initialization"]["text_embedding_sha256"]:
        raise ValueError("Text embeddings differ from training")
    metadata = {key: checkpoint[key] for key in ("completed_epochs", "planned_epochs", "training_complete",
                "global_step", "amp_skipped_updates", "config", "loss_spec", "initialization", "sources", "data_lock", "runtime")}
    metadata.update(checkpoint_sha256=sha256(path), model="I-MMSeg", label_order="canonical")
    return checkpoint, BenchmarkDataset(data_root, list_dir, "test_vol", label_order), metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--list-dir")
    parser.add_argument("--label-order", choices=("legacy", "canonical"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("none", "bf16"), default="none")
    args = parser.parse_args(argv)
    checkpoint, dataset, metadata = validate_run(args.checkpoint, args.data_root, args.list_dir, args.label_order)
    torch.set_num_threads(checkpoint["config"]["cpu_threads"])
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if args.batch_size < 1 or (device.type != "cuda" and args.amp != "none"):
        parser.error("Positive batch size required; AMP requires CUDA")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh evaluation output directory")
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dtype = {"none": None, "bf16": torch.bfloat16}[args.amp]
    metadata.update(evaluation_amp=args.amp, evaluation_device=str(device))
    prediction_dir = output / "test_predictions"
    prediction_dir.mkdir(parents=True)
    for sample in dataset:
        prediction = predict_volume(model, [sample[k] for k in ("image", "image1", "image2")],
                                    args.batch_size, device, dtype)
        np.savez_compressed(prediction_dir / f"{sample['case_name']}_pred.npz", prediction=prediction)
    metadata["prediction_sha256"] = {p.name: sha256(p) for p in sorted(prediction_dir.glob("*.npz"))}
    (output / "prediction_metadata.json").write_text(json.dumps(dict(metadata, inference_protocol=INFERENCE), indent=2), encoding="utf-8")
    result = score_predictions(prediction_dir, dataset, output / "report", metadata)
    print(f"Saved 76 canonical predictions and report: {output / 'report' / 'metrics.json'}")


if __name__ == "__main__":
    main()
