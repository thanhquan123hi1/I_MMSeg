"""Two optimization steps on train-only or synthetic inputs. No test access/checkpoint."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from benchmark.data import BenchmarkDataset
from benchmark.model import build_model
from benchmark.objective import OriginalObjective
from benchmark.protocol import REFERENCE
from benchmark.training import seed_everything, float_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="Optional normalized cache; only first TRAIN slice is opened")
    parser.add_argument("--label-order", choices=("legacy", "canonical"), default="legacy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", choices=("none", "bf16"), default="none")
    args = parser.parse_args()
    seed_everything(1234)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if args.amp == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("BF16 requires a CUDA device with BF16 support")
    if args.data_root:
        sample = BenchmarkDataset(args.data_root, REFERENCE / "splits", "train", args.label_order, augment=True)[0]
        images = [sample[k].unsqueeze(0).to(device) for k in ("image", "image1", "image2")]
        target = sample["label"].unsqueeze(0).to(device)
        source = sample["case_name"]
    else:
        images = [torch.rand(1, 1, 128, 128, device=device) for _ in range(3)]
        target = torch.zeros(1, 128, 128, dtype=torch.long, device=device)
        target[:, :64, 64:] = 1; target[:, 64:, :64] = 2; target[:, 64:, 64:] = 3
        source = "synthetic"
    model = build_model({"vit_name": "R50-ViT-B_16"}).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    objective = OriginalObjective()
    report = []
    for epoch in (0, 100):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp == "bf16"):
            outputs = model(*images, objective.do_contrast(epoch))
        with contextlib.redirect_stdout(io.StringIO()):
            loss = objective(float_outputs(outputs), target, epoch).mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError("Non-finite model/loss/gradient smoke check")
        optimizer.step()
        report.append(dict(epoch_index=epoch, loss=loss.item(), gradient_norm=norm.item(),
                           seconds=time.perf_counter() - started))
    print(json.dumps(dict(source=source, device=str(device), amp=args.amp, optimizer_steps=2, steps=report,
                          note="Software smoke check, not a training run or accuracy result"), indent=2))


if __name__ == "__main__":
    main()
