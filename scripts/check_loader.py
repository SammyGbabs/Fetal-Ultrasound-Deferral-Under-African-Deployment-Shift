"""T2 'done when' check: load one batch from every split and print its shape.

Both Samuel and Akachi run this in Colab after the template's setup cells:
    !python scripts/check_loader.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import make_loader  # noqa: E402

for split in ["spain_train", "spain_val", "spain_test", "africa_test"]:
    loader = make_loader(split, train=(split == "spain_train"), batch_size=16)
    batch = next(iter(loader))
    x, y = batch["image"], batch["label"]
    print(f"{split:12s} {len(loader.dataset):5d} images | batch {tuple(x.shape)} "
          f"| labels {sorted(set(y.tolist()))} | pixel range {x.min():.2f} to {x.max():.2f}")
print("\nLoader check passed.")
