"""MIDL: flag text and measurement markers burned into the test images.

Calipers drawn inside the fan can give the view away (e.g. "BPD"/"OFD" on head views,
"ATD"/"ASD" on abdomen views) and can't be masked cleanly. For every image of
spain_test and africa_test this:

- runs OCR (EasyOCR) on the original and splits each detection into inside or
  outside the fan, using src/masking.py,
- counts '+' caliper crosses inside the fan (src/markers.py), which OCR can't see,
- runs OCR again on the masked image, to catch any text the masking left behind.

It writes one CSV (default outputs/markers_ocr.csv on the shared Drive):

    image_id, split, has_marker, marker_text, caliper_crosses, outside_text,
    residual_text, fan_fraction

has_marker is True when OCR finds text inside the fan or at least one caliper cross is
found. Detections are joined with " | ". Run in Colab after the template's setup cells;
EasyOCR is not in requirements.txt, so install it first:

    !pip install -q easyocr
    !python scripts/find_markers.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.markers import caliper_crosses  # noqa: E402
from src.masking import image_mask  # noqa: E402
from src.paths import DRIVE_ROOT, data_path  # noqa: E402

SPLITS = ["spain_test", "africa_test"]
MIN_CONFIDENCE = 0.4    # EasyOCR confidence; speckle below this is mostly noise
EDGE_MARGIN = 15        # pixels; text this close to the fan edge counts as outside


def read_text(reader, gray: np.ndarray) -> list[tuple[str, int, int]]:
    """OCR detections above MIN_CONFIDENCE that contain a letter or digit, with their centre."""
    found = []
    for box, text, confidence in reader.readtext(gray):
        text = text.strip()
        if confidence >= MIN_CONFIDENCE and any(ch.isalnum() for ch in text):
            x, y = np.mean(box, axis=0).astype(int)
            found.append((text, min(x, gray.shape[1] - 1), min(y, gray.shape[0] - 1)))
    return found


def scan(reader, path: Path, source: str) -> dict:
    gray = np.asarray(Image.open(path).convert("L"))
    fan = image_mask(gray, source)          # the fan, minus the screen overlays
    inner = ndimage.binary_erosion(fan, iterations=EDGE_MARGIN)
    detections = read_text(reader, gray)
    inside = [t for t, x, y in detections if inner[y, x]]
    outside = [t for t, x, y in detections if not inner[y, x]]
    crosses = caliper_crosses(np.where(inner, gray, 0))
    residual = [t for t, _, _ in read_text(reader, np.where(fan, gray, 0).astype(np.uint8))]
    return {"has_marker": bool(inside) or crosses > 0, "marker_text": " | ".join(inside),
            "caliper_crosses": crosses, "outside_text": " | ".join(outside),
            "residual_text": " | ".join(residual), "fan_fraction": round(float(fan.mean()), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    parser.add_argument("--out", default=str(DRIVE_ROOT / "outputs" / "markers_ocr.csv"))
    args = parser.parse_args()

    import easyocr
    import torch
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    rows = []
    for split in args.splits:
        df = pd.read_csv(ROOT / "splits" / f"{split}.csv", keep_default_na=False)
        sources = df["machine"].where(df["source"] == "spain", df["country"])
        for n, (image_id, relative, source) in enumerate(zip(df["image_id"], df["path"], sources), 1):
            rows.append({"image_id": image_id, "split": split, **scan(reader, data_path(relative), source)})
            if n % 250 == 0:
                print(f"  {split}: {n}/{len(df)}", flush=True)
        done = [r for r in rows if r["split"] == split]
        print(f"ok  {split}: {len(df)} images | {sum(r['has_marker'] for r in done)} with a marker "
              f"inside the fan | {sum(bool(r['residual_text']) for r in done)} with text left after masking")
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
