"""Train I-MMSeg using the fixed MyoPS380 train304/test76, last-epoch protocol."""
import argparse
from pathlib import Path

import torch

from benchmark.data import BenchmarkDataset
from benchmark.model import build_model, initialization_record
from benchmark.protocol import REFERENCE, lock_data, source_record
from benchmark.training import load_checkpoint, run_training, seed_everything


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Normalized Processed_data containing bSSFP/LGE/T2w")
    parser.add_argument("--list-dir", "--list_dir", default=str(REFERENCE / "splits"))
    parser.add_argument("--label-order", required=True, choices=("legacy", "canonical"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-epochs", "--max_epochs", type=int, default=300)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=16,
                        help="Total physical batch on ONE device; not multiplied by GPU count")
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--base-lr", "--base_lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cuda:0 or cpu; single-device execution")
    parser.add_argument("--amp", choices=("none", "bf16"), default="none",
                        help="FP16 is unsupported: original cosine attention overflows in half precision")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--vit-name", "--vit_name", choices=("R50-ViT-B_16",), default="R50-ViT-B_16")
    parser.add_argument("--pretrained", help="Optional R50-ViT-B_16.npz; absent means RANDOM encoders, as in original entry point")
    parser.add_argument("--resume", help="Trusted full last.pth; reapply original CLI configuration")
    parser.add_argument("--epochs-per-run", type=int, default=0, help="Pause after N additional epochs without changing 300-epoch schedule")
    # Locked original objective; flags retained only to make the selected recipe explicit.
    parser.add_argument("--start-contrast-epoch", type=int, choices=(20,), default=20)
    parser.add_argument("--region-fusion-start-epoch", type=int, choices=(100,), default=100)
    parser.add_argument("--contrast-sample-num", type=int, choices=(10,), default=10)
    parser.add_argument("--contrast-w", type=float, choices=(0.1,), default=0.1)
    args = parser.parse_args(argv)
    if args.max_epochs != 300 or min(args.batch_size, args.accum_steps, args.cpu_threads) < 1:
        parser.error("Requires 300 epochs and positive batch/accumulation/thread counts")
    if args.num_workers < 0 or args.epochs_per_run < 0 or not 0 < args.base_lr < 1:
        parser.error("Invalid workers, pause budget or LR")
    if args.pretrained and args.resume:
        parser.error("Use --resume without --pretrained; checkpoint retains initialization provenance")
    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; select --device cpu")
    for name in ("data_root", "list_dir", "output_dir"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    if args.pretrained:
        args.pretrained = str(Path(args.pretrained).resolve())
    return vars(args)


def main(argv=None):
    config = parse_args(argv)
    torch.set_num_threads(config["cpu_threads"])
    seed_everything(config["seed"])
    data_lock = lock_data(config["data_root"], config["list_dir"], config["label_order"])
    sources = source_record()
    initialization = initialization_record(config)
    if config["resume"]:
        saved = load_checkpoint(config["resume"])
        if saved["initialization"]["text_embedding_sha256"] != initialization["text_embedding_sha256"]:
            raise ValueError("Text embedding files changed")
        initialization = saved["initialization"]
    dataset = BenchmarkDataset(config["data_root"], config["list_dir"], "train", config["label_order"], augment=True)
    model = build_model(config, initialize=not bool(config["resume"]))
    print(f"Physical batch={config['batch_size']}, accumulation={config['accum_steps']}; original batch-dependent objective retained", flush=True)
    print(run_training(model, dataset, config, data_lock, sources, initialization))


if __name__ == "__main__":
    main()
