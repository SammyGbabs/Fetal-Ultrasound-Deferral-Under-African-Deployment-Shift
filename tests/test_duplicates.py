"""Checks for scripts/find_duplicates.py on synthetic images."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.find_duplicates import find_duplicates, fingerprint  # noqa: E402

rng = np.random.default_rng(0)


def scan(h: int = 300, w: int = 400) -> np.ndarray:
    """A sector of smooth random 'anatomy' on black, the same fan shape every time."""
    y, x = np.mgrid[:h, :w]
    fan = (np.abs(np.degrees(np.arctan2(x - w / 2, y + 40))) < 35) & (np.hypot(x - w / 2, y + 40) < 330)
    blobs = np.asarray(Image.fromarray(rng.integers(0, 255, (12, 16), dtype=np.uint8)).resize((w, h), Image.BICUBIC))
    image = np.where(fan, np.clip(blobs.astype(int) + rng.integers(-10, 10, (h, w)), 20, 255), 0)
    return image.astype(np.uint8)


def test_find_duplicates():
    a, other = scan(), scan()
    resized = np.asarray(Image.fromarray(a).resize((360, 270), Image.BILINEAR))
    frame = np.clip(a.astype(int) + rng.integers(-3, 4, a.shape), 0, 255).astype(np.uint8)
    df = pd.DataFrame({
        "image_id": ["a", "a_copy", "a_resized", "a_frame", "other"],
        "split": ["spain_train", "spain_test", "africa_test", "spain_train", "spain_val"],
        "patient_id": ["P1", "P2", "P3", "P1", "P4"],
        "label": [0, 0, 0, 0, 0],
    })
    prints = [fingerprint(g) for g in (a, a.copy(), resized, frame, other)]
    duplicates, matched = find_duplicates(df, [p[0] for p in prints], np.array([p[1] for p in prints]),
                                          np.stack([p[2] for p in prints]))
    got = duplicates.set_index("image_id")
    assert set(got.index) == {"a_copy", "a_resized"}            # frame: same patient; other: different scan
    assert got.loc["a_copy", "match_type"] == "exact"
    assert got.loc["a_resized", "match_type"] == "near"
    assert (got.duplicate_of == "a").all()                       # original from the earliest split
    assert (got.merge_into_patient_id == "P1").all()
    assert "other" not in set(matched.a) | set(matched.b)
