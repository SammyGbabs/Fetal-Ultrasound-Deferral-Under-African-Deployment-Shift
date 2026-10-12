"""Plane classifiers shared by training (T4) and inference (T5).

Every model is a pretrained timm backbone with its own classifier removed, followed by one dropout layer and a 4-class linear head. The pooled
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
    # Priority 2, foundation model (self-supervised, LVD-1689M)
    "dinov3s": "vit_small_patch16_dinov3.lvd1689m",
}


def normalization(model: str) -> tuple[tuple, tuple]:
    """Mean and std the pretrained weights expect; not every model uses the ImageNet values."""
    cfg = timm.get_pretrained_cfg(BACKBONES[model])
    return tuple(cfg.mean), tuple(cfg.std)


class PlaneClassifier(nn.Module):
    def __init__(self, model: str, num_classes: int = 4, dropout: float = 0.2,
                 pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
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


def build_model(model: str, num_classes: int = 4, dropout: float = 0.2,
                pretrained: bool = True, freeze_backbone: bool = False) -> PlaneClassifier:
    if model not in BACKBONES:
        raise ValueError(f"Unknown model '{model}'; choose from {sorted(BACKBONES)}")
    return PlaneClassifier(model, num_classes, dropout, pretrained, freeze_backbone)
