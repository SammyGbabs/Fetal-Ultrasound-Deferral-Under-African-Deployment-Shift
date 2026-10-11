"""Plane classifiers shared by training (T4) and inference (T5).

Every model is an ImageNet-pretrained timm backbone with its own classifier
removed, followed by one dropout layer and a 4-class linear head. The pooled
backbone output is the "penultimate feature" of contract D. Dropout is an
explicit nn.Dropout placed only before the head, so MC dropout works the same
way in every architecture and never depends on a backbone's built-in dropout.
"""
import timm
import torch
from torch import nn

# Short model name (used in run IDs) -> pinned timm weights
BACKBONES = {
    "effb0": "efficientnet_b0.ra_in1k",
    "rn50": "resnet50.tv_in1k",      # the classic torchvision weights, the usual baseline
}


class PlaneClassifier(nn.Module):
    def __init__(self, model: str, num_classes: int = 4, dropout: float = 0.2,
                 pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(BACKBONES[model], pretrained=pretrained, num_classes=0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled penultimate-layer features, N x D."""
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
                pretrained: bool = True) -> PlaneClassifier:
    if model not in BACKBONES:
        raise ValueError(f"Unknown model '{model}'; choose from {sorted(BACKBONES)}")
    return PlaneClassifier(model, num_classes, dropout, pretrained)
