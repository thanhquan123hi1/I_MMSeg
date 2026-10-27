"""Fixed-budget train-only execution, atomic checkpointing and exact epoch-boundary resume."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import platform
import warnings

import numpy as np
import torch
import scipy
import h5py
from torch.utils.data import DataLoader

from .objective import LOSS_SPEC, OriginalObjective
from .protocol import EPOCHS, PROTOCOL, TRAIN_COUNT, validate_checkpoint


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def rng_state(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                loader=generator.get_state())


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    generator.set_state(state["loader"].cpu())


def atomic_save(checkpoint, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    # Full local training checkpoints contain Python/NumPy RNG state. Only load trusted files.
    return torch.load(path, map_location="cpu", weights_only=False)


def locked_config(config):
    return {k: v for k, v in config.items() if k not in {"resume", "output_dir", "epochs_per_run", "pretrained"}}


def runtime_record():
    return dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                scipy=scipy.__version__, h5py=h5py.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version())


def float_outputs(value):
    """Preserve the original FP32 loss math under optional BF16 model autocast."""
    if torch.is_tensor(value):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, (list, tuple)):
        return type(value)(float_outputs(item) for item in value)
    return value


def train_epoch(model, loader, objective, optimizer, scheduler, scaler, device, epoch,
                accum_steps=1, amp_dtype=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    group_samples = updates = skipped = samples = 0
    total_loss = 0.0
    if amp_dtype == torch.float16:
        raise ValueError("FP16 is unsupported by the original I-MMSeg cosine attention; use FP32 or BF16")
    if amp_dtype == torch.bfloat16 and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 requires a CUDA device with BF16 support")
    for index, batch in enumerate(loader):
        inputs = [batch[k].to(device) for k in ("image", "image1", "image2")]
        target = batch["label"].to(device)
        count = target.shape[0]
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            outputs = model(*inputs, objective.do_contrast(epoch))
        # The original objective was FP32. Do not run its contrastive exp/log in BF16.
        with torch.autocast(device_type=device.type, enabled=False):
            loss = objective(float_outputs(outputs), target, epoch).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch}, batch={index}")
        # Accumulation of the original microbatch objective; NOT equivalent to large-batch contrastive/Dice.
        scaler.scale(loss * count).backward()
        group_samples += count
        samples += count
        total_loss += float(loss.detach()) * count
        if (index + 1) % accum_steps == 0 or index + 1 == len(loader):
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(group_samples)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm) and not scaler.is_enabled():
                raise FloatingPointError("Non-finite gradient norm")
            if not torch.isfinite(norm) and scaler.is_enabled() and all(
                    torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError("Gradient norm overflow despite finite gradient entries")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                skipped += 1
            else:
                scheduler.step()
                updates += 1
            optimizer.zero_grad(set_to_none=True)
            group_samples = 0
    if samples != len(loader.dataset) or not updates:
        raise RuntimeError("Incomplete train epoch or no successful optimizer updates")
    return dict(loss=total_loss / samples, samples=samples, updates=updates, skipped=skipped)


def run_training(model, dataset, config, data_lock, sources, initialization):
    if config["max_epochs"] != EPOCHS or len(dataset) != TRAIN_COUNT:
        raise ValueError("Official runs require 300 planned epochs and all 1972 slices")
    device = torch.device(config["device"])
    if config["amp"] not in {"none", "bf16"}:
        raise ValueError("FP16 is unsupported by the original model; use --amp none or bf16")
    dtype = {"none": None, "bf16": torch.bfloat16}[config["amp"]]
    if device.type != "cuda" and dtype is not None:
        raise ValueError("AMP is supported only on CUDA in this protocol")
    if config["accum_steps"] > 1:
        warnings.warn("I-MMSeg Dice/contrastive microbatch losses are NOT equivalent to a physical batch of 16")
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=False,
                        num_workers=config["num_workers"], worker_init_fn=seed_worker,
                        generator=generator, pin_memory=device.type == "cuda", persistent_workers=False)
    model.to(device)
    objective = OriginalObjective(config["contrast_w"], config["contrast_sample_num"],
                                  config["start_contrast_epoch"], config["region_fusion_start_epoch"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["base_lr"], weight_decay=1e-4)
    total_updates = EPOCHS * math.ceil(len(loader) / config["accum_steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0., 1. - step / total_updates) ** 0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    start = global_step = skipped_total = 0
    runtime = runtime_record()
    if config["resume"]:
        checkpoint = validate_checkpoint(load_checkpoint(config["resume"]))
        if (locked_config(checkpoint["config"]) != locked_config(config)
                or checkpoint["data_lock"] != data_lock
                or checkpoint["sources"]["files_sha256"] != sources["files_sha256"]
                or checkpoint["loss_spec"] != LOSS_SPEC
                or checkpoint.get("runtime") != runtime
                or checkpoint["initialization"] != initialization):
            raise ValueError("Resume configuration/data/source/initialization differs from saved run")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"], generator)
        start = checkpoint["completed_epochs"]
        global_step, skipped_total = checkpoint["global_step"], checkpoint["amp_skipped_updates"]
    output = Path(config["output_dir"])
    if not config["resume"] and output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory for a new run")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("best.*")):
        raise ValueError("No-val output directory must not contain best checkpoints")
    if config["resume"] and Path(config["resume"]).resolve().parent != output.resolve():
        raise ValueError("Resume into the checkpoint's original run directory")
    (output / "config.json").write_text(json.dumps(dict(config=config, data_lock=data_lock,
            sources=sources, initialization=initialization, runtime=runtime, loss_spec=LOSS_SPEC), indent=2), encoding="utf-8")
    manifests = output / "splits"
    manifests.mkdir(exist_ok=True)
    for name in ("train", "test_vol"):
        content = (Path(config["list_dir"]) / f"{name}.txt").read_bytes()
        saved = manifests / f"{name}.txt"
        if saved.exists() and saved.read_bytes() != content:
            raise ValueError("Saved run manifest was modified")
        saved.write_bytes(content)
    stop = min(EPOCHS, start + config["epochs_per_run"]) if config["epochs_per_run"] else EPOCHS
    log_path = output / "train.jsonl"
    for epoch in range(start, stop):
        metrics = train_epoch(model, loader, objective, optimizer, scheduler, scaler, device,
                              epoch, config["accum_steps"], dtype)
        global_step += metrics["updates"]
        skipped_total += metrics["skipped"]
        checkpoint = dict(protocol=PROTOCOL, epoch=epoch, planned_epochs=EPOCHS,
            completed_epochs=epoch + 1, training_complete=epoch + 1 == EPOCHS,
            checkpoint_selection="last_completed_epoch", checkpoint_selection_metric=None,
            model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(), rng=rng_state(generator), config=config, loss_spec=LOSS_SPEC,
            initialization=initialization, sources=sources, data_lock=data_lock,
            runtime=runtime,
            total_updates=total_updates, global_step=global_step, amp_skipped_updates=skipped_total)
        validate_checkpoint(checkpoint)
        atomic_save(checkpoint, output / "last.pth")
        row = dict(epoch=epoch + 1, global_step=global_step, lr=optimizer.param_groups[0]["lr"], **metrics)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    return output / "last.pth"
