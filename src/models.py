"""Plane classifiers shared by training (T4) and inference (T5).

Every model is a pretrained image encoder with its own classifier removed,
followed by one dropout layer and a 4-class linear head. All encoders come
from timm except FetalCLIP, which is loaded through open_clip. The pooled
backbone output is the "penultimate feature" of contract D. Dropout is an
explicit nn.Dropout placed only before the head, so MC dropout works the same
way in every architecture and never depends on a backbone's built-in dropout.
"""
import timm
import torch
from torch import nn

# Short model name (used in run IDs) -> pinned timm weights
BACKBONES = {
    # Priority 1
    "effb0": "efficientnet_b0.ra_in1k",
    "rn50": "resnet50.tv_in1k",      # the classic torchvision weights, the usual baseline
    # Priority 2, ImageNet-1k
    "mnv4": "mobilenetv4_conv_medium.e500_r224_in1k",
    "convnext": "convnext_tiny.fb_in1k",
    "vits16": "vit_small_patch16_224.augreg_in1k",
    # Priority 2, foundation models
    "dinov3s": "vit_small_patch16_dinov3.lvd1689m",        # self-supervised, LVD-1689M
    "fetalclip": "numansaeed/fetalclip-model",             # fetal ultrasound CLIP, ViT-L/14
}

# FetalCLIP is not in timm. Its release is an open_clip checkpoint (CC BY-NC 4.0) plus
# this config, copied from FetalCLIP_config.json in github.com/BioMedIA-MBZUAI/FetalCLIP.
# Only the image encoder is used; its 768-d image embedding is the feature.
FETALCLIP_WEIGHTS = ("numansaeed/fetalclip-model", "FetalCLIP_weights.pt")
FETALCLIP_CONFIG = {
    "embed_dim": 768,
    "vision_cfg": {"image_size": 224, "layers": 24, "width": 1024, "patch_size": 14},
    "text_cfg": {"context_length": 117, "vocab_size": 49408, "width": 768, "heads": 12, "layers": 12},
}


def normalization(model: str) -> tuple[tuple, tuple]:
    """Mean and std the pretrained weights expect; not every model uses the ImageNet values."""
    if model == "fetalclip":
        from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
        return tuple(OPENAI_DATASET_MEAN), tuple(OPENAI_DATASET_STD)
    cfg = timm.get_pretrained_cfg(BACKBONES[model])
    return tuple(cfg.mean), tuple(cfg.std)


def fetalclip_encoder(pretrained: bool) -> nn.Module:
    import open_clip
    from huggingface_hub import hf_hub_download
    open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = FETALCLIP_CONFIG
    weights = hf_hub_download(*FETALCLIP_WEIGHTS) if pretrained else None
    encoder = open_clip.create_model("FetalCLIP", pretrained=weights).visual
    encoder.head_hidden_size = FETALCLIP_CONFIG["embed_dim"]
    return encoder


class PlaneClassifier(nn.Module):
    def __init__(self, model: str, num_classes: int = 4, dropout: float = 0.2,
                 pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        if model == "fetalclip":
            self.backbone = fetalclip_encoder(pretrained)
        else:
            self.backbone = timm.create_model(BACKBONES[model], pretrained=pretrained, num_classes=0)
        self.dropout = nn.Dropout(dropout)
        # head_hidden_size is the width of the pooled output (differs from num_features in MobileNetV4)
        self.head = nn.Linear(self.backbone.head_hidden_size, num_classes)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def train(self, mode: bool = True):
        """A frozen backbone stays in eval mode, so only the linear head is ever trained."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled penultimate-layer features, N x D."""
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_backbone):
            return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.features(x)))

    @torch.no_grad()
    def predict(self, x: torch.Tensor, mc_passes: int = 20) -> dict:
        """Everything contract D needs for one batch, with the model in eval mode.

        logits:    N x C, dropout off
        features:  N x D
        mc_logits: N x mc_passes x C, dropout on. Batch norm stays in eval mode
                   because only the dropout layer is sampled, and since dropout sits
                   after the backbone the passes reuse one set of features.
        """
        self.eval()
        features = self.features(x).float()
        logits = self.head(features)
        mc_logits = torch.stack(
            [self.head(nn.functional.dropout(features, self.dropout.p, training=True))
             for _ in range(mc_passes)], dim=1)
        return {"logits": logits, "features": features, "mc_logits": mc_logits}


def saved_state(model: PlaneClassifier) -> dict:
    """Weights to checkpoint. A frozen backbone never changes, so only the head is saved
    and the backbone is rebuilt from its pretrained weights when loading."""
    state = model.state_dict()
    if model.freeze_backbone:
        state = {k: v for k, v in state.items() if not k.startswith("backbone.")}
    return state


def load_state(model: PlaneClassifier, state: dict) -> None:
    """Inverse of saved_state. Also loads full checkpoints of frozen runs."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or (missing and not model.freeze_backbone) \
            or any(not k.startswith("backbone.") for k in missing):
        raise RuntimeError(f"Checkpoint does not match the model: missing {missing[:5]}, "
                           f"unexpected {unexpected[:5]}")


def build_model(model: str, num_classes: int = 4, dropout: float = 0.2,
                pretrained: bool = True, freeze_backbone: bool = False) -> PlaneClassifier:
    if model not in BACKBONES:
        raise ValueError(f"Unknown model '{model}'; choose from {sorted(BACKBONES)}")
    return PlaneClassifier(model, num_classes, dropout, pretrained, freeze_backbone)
