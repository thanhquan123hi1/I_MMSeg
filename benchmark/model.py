"""Official architecture; only initialization/device plumbing is adapted."""
import copy
import numpy as np
import torch

from .protocol import ROOT, sha256


def load_encoder_weights(model, path):
    """Load only the three actual ResNetV2 encoders; legacy load_from targets absent ViT modules."""
    assignments = []
    with np.load(path, allow_pickle=False) as weights:
        for encoder in (model.transformer1, model.transformer2, model.transformer3):
            body = encoder.embeddings.hybrid_model
            params = [("conv_root/kernel", body.root.conv.weight, True),
                      ("gn_root/scale", body.root.gn.weight, False),
                      ("gn_root/bias", body.root.gn.bias, False)]
            for block_name, block in body.body.named_children():
                for unit_name, unit in block.named_children():
                    prefix = f"{block_name}/{unit_name}"
                    for i in (1, 2, 3):
                        params.extend([(f"{prefix}/conv{i}/kernel", getattr(unit, f"conv{i}").weight, True),
                                       (f"{prefix}/gn{i}/scale", getattr(unit, f"gn{i}").weight, False),
                                       (f"{prefix}/gn{i}/bias", getattr(unit, f"gn{i}").bias, False)])
                    if hasattr(unit, "downsample"):
                        params.extend([(f"{prefix}/conv_proj/kernel", unit.downsample.weight, True),
                                       (f"{prefix}/gn_proj/scale", unit.gn_proj.weight, False),
                                       (f"{prefix}/gn_proj/bias", unit.gn_proj.bias, False)])
            for key, param, conv in params:
                value = weights[key]
                value = value.transpose(3, 2, 0, 1) if conv else value.reshape(-1)
                if value.shape != tuple(param.shape) or not np.isfinite(value).all():
                    raise ValueError(f"Incompatible/non-finite pretrained tensor: {key}")
                assignments.append((param, torch.from_numpy(value.copy())))
    with torch.no_grad():
        for param, value in assignments:
            param.copy_(value)
    return len(assignments)


def build_model(config, initialize=False):
    from networks.vit_seg_modeling import VisionTransformer, CONFIGS
    model_config = copy.deepcopy(CONFIGS[config["vit_name"]])
    model_config.n_classes = 4
    model_config.n_skip = 3
    model_config.patches.grid = (8, 8)
    model = VisionTransformer(model_config, img_size=128, num_classes=4)
    if initialize and config.get("pretrained"):
        load_encoder_weights(model, config["pretrained"])
    return model


def initialization_record(config):
    paths = [ROOT / "text_features" / name for name in
             ("embedding_class_information.pth", "embedding_MRI_information.pth")]
    record = {"encoder": "random", "text_embedding_sha256": {p.name: sha256(p) for p in paths},
              "text_label_order": "legacy"}
    if config.get("pretrained"):
        record.update(encoder="R50-ViT-B_16.npz ResNetV2 weights in all three modality encoders",
                      encoder_sha256=sha256(config["pretrained"]))
    return record
