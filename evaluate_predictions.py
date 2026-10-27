"""Shared pinned SCAR metric evaluator for canonical I-MMSeg or M2-Plus exports."""
import argparse
import json
from pathlib import Path

from benchmark.data import BenchmarkDataset, ORDERS
from benchmark.evaluation import score_predictions
from benchmark.protocol import PROTOCOL, TRAIN_COUNT, lock_data, sha256
from benchmark.training import load_checkpoint
from evaluate import validate_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=("I-MMSeg", "M2-Plus"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--list-dir")
    parser.add_argument("--label-order", choices=("legacy", "canonical"))
    args = parser.parse_args(argv)
    if args.model == "I-MMSeg":
        _, dataset, metadata = validate_run(args.checkpoint, args.data_root, args.list_dir, args.label_order)
        sidecar = Path(args.predictions).parent / "prediction_metadata.json"
        if not sidecar.is_file():
            raise ValueError("Missing prediction provenance from evaluate.py")
        provenance = json.loads(sidecar.read_text())
        if provenance["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
            raise ValueError("Predictions belong to another checkpoint")
        metadata = provenance
    else:
        # No model is constructed here. SCAR full checkpoint metadata is checked independently.
        checkpoint = load_checkpoint(args.checkpoint)
        config = checkpoint["args"]
        if (Path(args.checkpoint).name != "last.pt" or checkpoint.get("training_protocol", {}).get("id") != PROTOCOL
                or not config.get("no_val") or config.get("max_epochs") != 300
                or checkpoint.get("completed_epochs") != 300 or checkpoint.get("planned_epochs") != 300
                or checkpoint.get("epoch") != 299 or checkpoint.get("training_complete") is not True):
            raise ValueError("M2-Plus checkpoint is not a completed no-val 300-epoch run")
        batches = (TRAIN_COUNT + config["batch_size"] - 1) // config["batch_size"]
        per_epoch = (batches + config["accum_steps"] - 1) // config["accum_steps"]
        if (checkpoint["total_updates"] != 300 * per_epoch or checkpoint["global_step"] < 1
                or checkpoint["global_step"] + checkpoint["amp_skipped_updates"] != 300 * per_epoch
                or checkpoint["scheduler"]["last_epoch"] != checkpoint["global_step"]
                or checkpoint["model_config"]["ablation"].upper() != "M2-PLUS"):
            raise ValueError("Inconsistent M2-Plus architecture/update budget")
        root = args.data_root or config["data_root"]
        lists = args.list_dir or config["list_dir"]
        order = args.label_order or config["label_order"]
        data_lock = lock_data(root, lists, order)
        saved = checkpoint["benchmark_data"]
        if ORDERS[order] != ORDERS.get(saved.get("source_label_order")):
            raise ValueError("M2-Plus source label order differs from its checkpoint")
        if any(data_lock[k] != saved[k] for k in ("cache_sha256", "split_ids_sha256")):
            raise ValueError("M2-Plus evaluation cache/split mismatch")
        sidecar = Path(args.predictions) / "metrics.json"
        if not sidecar.is_file():
            raise ValueError("Expected the original SCAR evaluation directory, including metrics.json provenance")
        provenance = json.loads(sidecar.read_text())
        if (provenance["checkpoint_sha256"] != sha256(args.checkpoint)
                or provenance["inference_protocol"]["id"] != "single_pass_argmax_v1"):
            raise ValueError("SCAR prediction provenance/checkpoint mismatch")
        dataset = BenchmarkDataset(root, lists, "test_vol", order)
        metadata = dict(model="M2-Plus", checkpoint_sha256=sha256(args.checkpoint),
                        completed_epochs=300, planned_epochs=300, training_complete=True,
                        config=config, data_lock=data_lock, label_order="canonical",
                        initialization=checkpoint.get("initialization"),
                        sources=checkpoint.get("code_provenance"),
                        loss_protocol_version=checkpoint.get("loss_protocol_version"),
                        note="Metrics recomputed from supplied SCAR canonical predictions; no model inference performed")
    result = score_predictions(args.predictions, dataset, args.output_dir, metadata)
    print(f"Saved shared evaluation report: {Path(args.output_dir) / 'metrics.json'}")


if __name__ == "__main__":
    main()
