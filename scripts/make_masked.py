"""MIDL: build the masked test sets, with everything outside the ultrasound fan blacked out.

Burned-in text (scanner name, exam preset, depth, gain) sits outside the fan, so a
model could read it instead of the anatomy. For spain_val, spain_test and africa_test
this masks each original full-resolution image with src/masking.py (outside the fan,
plus the scanner's fixed screen overlays) and saves it
losslessly as PNG under the same relative path, so src/data.py finds each copy by
swapping "data/raw" for the condition's image_root, as for the compressed sets.

Images are written to this machine's local disk, then each condition is packed into one
.tar on the shared Drive (data/masked/), with a CSV of how much of each image the fan
covers, and a preview grid is saved. The three conditions are already listed in
splits/conditions.csv (spain_val_masked, spain_test_masked, africa_test_masked). Run
once, in Colab, after the template's setup cells:
    !python scripts/make_masked.py
In later sessions the template unpacks the tars.
"""
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.masking import mask_outside_fan  # noqa: E402
from src.paths import DRIVE_ROOT, LOCAL_ROOT, data_path  # noqa: E402

SPLITS = ["spain_val", "spain_test", "africa_test"]


def process_one(job: tuple) -> dict:
    image_id, relative, image_root, source = job
    target = LOCAL_ROOT / relative.replace("data/raw", image_root, 1)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(data_path(relative)) as image:
        masked, mask = mask_outside_fan(image, source)
    masked.save(target, format="PNG")
    return {"image_id": image_id, "fan_fraction": round(float(mask.mean()), 4)}


def build_condition(split: str) -> None:
    df = pd.read_csv(ROOT / "splits" / f"{split}.csv", keep_default_na=False)
    condition_id = f"{split}_masked"
    image_root = f"data/masked/{condition_id}"
    # Overlay positions depend on the scanner (Spain) or the centre (Africa)
    sources = df["machine"].where(df["source"] == "spain", df["country"])
    jobs = [(i, p, image_root, s) for i, p, s in zip(df["image_id"], df["path"], sources)]
    with ProcessPoolExecutor() as pool:
        stats = pd.DataFrame(list(pool.map(process_one, jobs, chunksize=16)))

    out_dir = DRIVE_ROOT / "data/masked"
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_dir / f"{condition_id}.tar", "w") as tar:
        tar.add(LOCAL_ROOT / image_root, arcname=image_root)
    stats.to_csv(out_dir / f"{condition_id}_fan.csv", index=False)

    # A fan covering very little or nearly all of the image means detection went wrong
    odd = stats[(stats.fan_fraction < 0.2) | (stats.fan_fraction > 0.95)]
    print(f"ok  {condition_id:20s} {len(df):5d} images | fan covers "
          f"{stats.fan_fraction.median():.0%} of the image (median) | "
          f"{len(odd)} images outside 20-95%" + (f": {', '.join(odd.image_id[:5])}" if len(odd) else ""))


def preview_grid(width: int = 320) -> None:
    """Original next to masked, one test image per Spanish machine and per African country."""
    rows = []
    spain = pd.read_csv(ROOT / "splits/spain_test.csv", keep_default_na=False)
    for machine, group in spain.groupby("machine"):
        rows.append((f"Spain, {machine}", group.iloc[0].path, "data/masked/spain_test_masked"))
    africa = pd.read_csv(ROOT / "splits/africa_test.csv", keep_default_na=False)
    for country, group in africa.groupby("country"):
        rows.append((country, group.iloc[0].path, "data/masked/africa_test_masked"))

    pairs = []
    for label, path, root in rows:
        images = []
        for p in (path, path.replace("data/raw", root, 1)):
            image = Image.open(data_path(p)).convert("L")
            image.thumbnail((width, width))
            images.append(image)
        pairs.append((label, images))
    label_height = 18
    height = sum(label_height + max(im.size[1] for im in images) + 10 for _, images in pairs)
    grid = Image.new("L", (2 * width + 10, height + label_height), 255)
    draw = ImageDraw.Draw(grid)
    draw.text((5, 3), "original", fill=0)
    draw.text((width + 15, 3), "masked", fill=0)
    top = label_height
    for label, images in pairs:
        draw.text((5, top + 3), label, fill=0)
        for j, image in enumerate(images):
            grid.paste(image, (j * (width + 10), top + label_height))
        top += label_height + max(im.size[1] for im in images) + 10
    target = DRIVE_ROOT / "data/masked/preview_grid.png"
    grid.save(target)
    print(f"ok  preview grid saved to {target}")


def main() -> None:
    for split in SPLITS:
        build_condition(split)
    preview_grid()


if __name__ == "__main__":
    main()
