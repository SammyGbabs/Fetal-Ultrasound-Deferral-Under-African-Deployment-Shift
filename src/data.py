"""Image loading shared by training (T4), inference (T5) and compression (T3).

Every split CSV (contract A) lists images with paths relative to the project
root. FetalPlanes reads any of them, optionally redirecting paths to a
compressed copy (contract B), and returns tensors ready for the model.
"""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.paths import data_path

ROOT = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(train: bool, image_size: int = 224,
                    mean=IMAGENET_MEAN, std=IMAGENET_STD) -> transforms.Compose:
    """Resize to a square, convert grayscale to 3 channels, normalize.

    Training adds mild augmentation only: small rotations, shifts, zoom and
    brightness/contrast changes. No JPEG or blur augmentation, which would
    contaminate the compression experiment, and no flips.
    """
    steps = [transforms.Resize((image_size, image_size))]
    if train:
        steps += [
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    steps += [
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(steps)


class FetalPlanes(Dataset):
    """One split CSV as a dataset.

    split:        name of a CSV in splits/ (e.g. "spain_train") or a path to one
    transform:    from build_transform
    image_root:   None for original images, or e.g. "data/compressed/africa_test_q30"
                  to read the same images from a compressed copy; it replaces the
                  leading "data/raw" of every path
    """

    def __init__(self, split: str, transform, image_root: str | None = None):
        csv = Path(split) if split.endswith(".csv") else ROOT / "splits" / f"{split}.csv"
        self.df = pd.read_csv(csv, keep_default_na=False)
        self.transform = transform
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.df)

    def resolve(self, relative: str) -> Path:
        if self.image_root:
            relative = relative.replace("data/raw", self.image_root, 1)
        return data_path(relative)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        image = Image.open(self.resolve(row["path"])).convert("L")
        return {
            "image": self.transform(image),
            "label": torch.tensor(int(row["label"])),
            "index": i,                      # row number, maps back to image_id
        }


def make_loader(split: str, train: bool, batch_size: int = 64, num_workers: int = 2,
                image_size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                image_root: str | None = None, seed: int = 0) -> DataLoader:
    """Training loaders shuffle; every other loader keeps CSV row order (contract D)."""
    dataset = FetalPlanes(split, build_transform(train, image_size, mean, std), image_root)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=False, generator=generator)
