"""Canonical predictions, batched single-pass inference, pinned SCAR metrics."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import convert_labels
from .protocol import PROTOCOL, sha256
from .scar_metrics import BENCHMARK_PROTOCOL, dataset_rows, summarize_dataset_rows

INFERENCE = dict(id="single_pass_argmax_v1", image_resize="bilinear_align_corners_false",
                 logit_resize="bilinear_align_corners_false", prediction_rule="argmax_logits",
                 mask_processing="none", forward_passes_per_batch=1)


@torch.inference_mode()
def predict_volume(model, images, batch_size=8, device=None, amp_dtype=None):
    arrays = [np.asarray(x, dtype=np.float32) for x in images]
    if (len(arrays) != 3 or any(x.ndim != 3 or x.shape != arrays[0].shape for x in arrays)
            or any(not np.isfinite(x).all() for x in arrays) or batch_size < 1):
        raise ValueError("Expected three aligned finite HWD volumes and positive inference batch")
    device = torch.device(device or next(model.parameters()).device)
    if amp_dtype == torch.float16:
        raise ValueError("FP16 is unsupported by the original cosine attention; use FP32 or BF16")
    if amp_dtype == torch.bfloat16 and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 requires a CUDA device with BF16 support")
    height, width, depth = arrays[0].shape
    prediction = np.empty((height, width, depth), dtype=np.uint8)
    was_training = model.training
    model.eval()
    try:
        for start in range(0, depth, batch_size):
            stop = min(start + batch_size, depth)
            inputs = [torch.from_numpy(np.ascontiguousarray(x[:, :, start:stop].transpose(2, 0, 1)))
                      .unsqueeze(1).to(device) for x in arrays]
            inputs = [F.interpolate(x, (128, 128), mode="bilinear", align_corners=False)
                      if x.shape[-2:] != (128, 128) else x for x in inputs]
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(*inputs)
            if logits.shape[:2] != (stop - start, 4) or not torch.isfinite(logits).all():
                raise ValueError("Invalid I-MMSeg logits")
            if logits.shape[-2:] != (height, width):
                logits = F.interpolate(logits.float(), (height, width), mode="bilinear", align_corners=False)
            legacy = logits.argmax(1).cpu().numpy().transpose(1, 2, 0)
            prediction[:, :, start:stop] = convert_labels(legacy, "legacy", "canonical")
    finally:
        model.train(was_training)
    return prediction


def score_predictions(prediction_dir, dataset, output_dir, metadata):
    prediction_dir, output_dir = Path(prediction_dir), Path(output_dir)
    expected = {f"{name}_pred.npz" for name in dataset.names}
    actual = {p.name for p in prediction_dir.glob("*.npz")}
    if actual != expected or len(expected) != 76:
        raise ValueError("Prediction files must match exactly the 76 locked CaseIDs")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Use an empty report directory")
    hashes = {name: sha256(prediction_dir / name) for name in sorted(expected)}
    if metadata.get("prediction_sha256") is not None and metadata["prediction_sha256"] != hashes:
        raise ValueError("Prediction files differ from the inference provenance")
    rows = []
    for sample in dataset:
        case, target = sample["case_name"], sample["label"]
        with np.load(prediction_dir / f"{case}_pred.npz", allow_pickle=False) as data:
            pred = data["prediction"]
            if (pred.dtype != np.uint8 or pred.ndim != 3 or pred.shape != target.shape
                    or not np.isin(pred, [0, 1, 2, 3]).all()):
                raise ValueError(f"Invalid canonical HWD uint8 prediction for {case}")
            rows.extend(dataset_rows(pred, target, case, dataset_id="myops380"))
    protocol = dict(BENCHMARK_PROTOCOL)
    protocol.pop("selection_metric", None)
    result = dict(metadata)
    # Provenance must never overwrite freshly computed metrics.
    result.update(summarize_dataset_rows(rows, dataset_id="myops380"))
    result.update(protocol=PROTOCOL, benchmark_protocol=protocol,
                  checkpoint_selection="last_completed_epoch", checkpoint_selection_metric=None,
                  inference_protocol=INFERENCE, case_count=76, prediction_sha256=hashes,
                  )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_case.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result
