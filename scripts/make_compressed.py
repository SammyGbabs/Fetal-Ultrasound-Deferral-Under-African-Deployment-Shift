"""T3: build the compressed test sets (contract B).

For spain_test and africa_test, simulates tele-ultrasound transmission:
each image is JPEG-encoded at a given quality, decoded again, and saved
losslessly as PNG under the same relative path. The model therefore sees
exactly what a receiver would see, and src/data.py finds each copy by
swapping "data/raw" for the condition's image_root.

Conditions: JPEG quality 95, 75, 50, 30, 20, 10, plus a "messaging" profile
(longest side shrunk to 1,280 px, then JPEG quality 70; an approximation of
messaging-app transmission).

Images are written to this machine's local disk (fast), then each condition is
packed into one .tar on the shared Drive. Also writes splits/conditions.csv and
a preview grid. Run once, in Colab, after the template's setup cells:
    !python scripts/make_compressed.py
Then commit splits/conditions.csv to Git.
"""
import io
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import DRIVE_ROOT, LOCAL_ROOT, data_path  # noqa: E402

TEST_SPLITS = ["spain_test", "africa_test"]
CLEAN_SPLITS = ["spain_train", "spain_val", "spain_test", "africa_test"]
JPEG_QUALITIES = [95, 75, 50, 30, 20, 10]
MESSAGING = {"max_side": 1280, "quality": 70}


def profiles() -> list[dict]:
    out = [{"name": f"q{q}", "compression": "jpeg", "quality": q, "max_side": None}
           for q in JPEG_QUALITIES]
    out.append({"name": "messaging", "compression": "messaging",
                "quality": MESSAGING["quality"], "max_side": MESSAGING["max_side"]})
    return out


def compress(image: Image.Image, quality: int, max_side: int | None) -> tuple[Image.Image, int]:
    """Return the decoded image after JPEG round trip, and the JPEG size in bytes."""
    if image.mode not in ("L", "RGB"):
        image = image.convert("RGB")
    if max_side and max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    size = buffer.tell()
    buffer.seek(0)
    return Image.open(buffer).copy(), size


def process_one(job: tuple) -> int:
    relative, image_root, quality, max_side = job
    source = data_path(relative)
    target = LOCAL_ROOT / relative.replace("data/raw", image_root, 1)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        decoded, size = compress(image, quality, max_side)
    decoded.save(target, format="PNG")
    return size


def build_condition(split: str, profile: dict) -> dict:
    df = pd.read_csv(ROOT / "splits" / f"{split}.csv")
    condition_id = f"{split}_{profile['name']}"
    image_root = f"data/compressed/{condition_id}"
    jobs = [(p, image_root, profile["quality"], profile["max_side"]) for p in df["path"]]
    with ProcessPoolExecutor() as pool:
        sizes = list(pool.map(process_one, jobs, chunksize=32))

    # Pack the condition into one tar on Drive (one large file syncs far faster than thousands)
    tar_path = DRIVE_ROOT / "data/compressed" / f"{condition_id}.tar"
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tar:
        tar.add(LOCAL_ROOT / image_root, arcname=image_root)

    mean_kb = sum(sizes) / len(sizes) / 1024
    print(f"ok  {condition_id:28s} {len(df):5d} images  mean {mean_kb:7.1f} KB as JPEG")
    return {"condition_id": condition_id, "split_file": f"{split}.csv",
            "compression": profile["compression"], "quality": profile["quality"],
            "image_root": image_root, "mean_kb_per_image": round(mean_kb, 1)}


def clean_condition(split: str) -> dict:
    df = pd.read_csv(ROOT / "splits" / f"{split}.csv")
    mean_kb = df["path"].map(lambda p: data_path(p).stat().st_size).mean() / 1024
    return {"condition_id": split, "split_file": f"{split}.csv", "compression": "none",
            "quality": "", "image_root": "", "mean_kb_per_image": round(mean_kb, 1)}


def preview_grid(conditions: pd.DataFrame, thumb: int = 180) -> None:
    """One African image per plane (rows) at every compression level (columns)."""
    df = pd.read_csv(ROOT / "splits" / "africa_test.csv")
    examples = df.groupby("label_name").head(1).sort_values("label")
    cols = [("original", "")] + [
        (c.condition_id.replace("africa_test_", ""), c.image_root)
        for c in conditions.itertuples() if c.condition_id.startswith("africa_test_")
    ]
    grid = Image.new("L", (thumb * len(cols), (thumb + 20) * len(examples) + 20), 255)
    draw = ImageDraw.Draw(grid)
    for j, (label, _) in enumerate(cols):
        draw.text((j * thumb + 5, 3), label, fill=0)
    for i, row in enumerate(examples.itertuples()):
        for j, (_, root) in enumerate(cols):
            path = row.path if not root else row.path.replace("data/raw", root, 1)
            image = Image.open(data_path(path)).convert("L")
            image.thumbnail((thumb, thumb))
            grid.paste(image, (j * thumb, 20 + i * (thumb + 20)))
        draw.text((5, 20 + i * (thumb + 20) + thumb + 2), row.label_name, fill=0)
    target = DRIVE_ROOT / "data/compressed/preview_grid.png"
    grid.save(target)
    print(f"ok  preview grid saved to {target}")


def main() -> None:
    rows = [clean_condition(s) for s in CLEAN_SPLITS]
    for split in TEST_SPLITS:
        for profile in profiles():
            rows.append(build_condition(split, profile))
    conditions = pd.DataFrame(rows)
    conditions.to_csv(ROOT / "splits" / "conditions.csv", index=False)
    print(f"\nWrote {len(conditions)} conditions to splits/conditions.csv. Commit it to Git.")
    preview_grid(conditions)


if __name__ == "__main__":
    main()