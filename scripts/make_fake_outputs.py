"""Write fake model outputs in the exact contract D format, for testing T6/T7 code.

The fake models are deliberately imperfect: accurate on Spanish images, worse
on African images, and worse again under heavy compression, so every metric
has something to measure.

    python scripts/make_fake_outputs.py /tmp/fake_outputs
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval import io  # noqa: E402

VARIANTS = {"fakecnn": 64, "fakevit": 32}      # variant -> feature dimension
SEEDS = range(5)


def noise_level(condition_id: str) -> float:
    level = 0.6 if condition_id.startswith("africa") else 0.25
    if "_q10" in condition_id:
        level += 0.4
    elif "_q20" in condition_id:
        level += 0.2
    return level


def fake(split: pd.DataFrame, condition_id: str, dim: int, rng) -> dict:
    y = split["label"].to_numpy()
    n = len(y)
    s = noise_level(condition_id)
    logits = rng.normal(0, 1.5 + 4 * s, (n, 4))
    logits[np.arange(n), y] += 5.0
    centres = np.random.default_rng(0).normal(0, 3, (4, dim))   # same class centres every run
    shift = 2.0 if condition_id.startswith("africa") else 0.0
    features = centres[y] + rng.normal(shift, 1 + s, (n, dim))
    mc = logits[:, None, :] + rng.normal(0, 0.5 + s, (n, io.MC_PASSES, 4))
    return {"image_id": np.array(split["image_id"].astype(str).tolist()), "y_true": y,
            "logits": logits.astype(np.float32), "features": features.astype(np.float32),
            "mc_logits": mc.astype(np.float32)}


def main(out_root: Path) -> None:
    conditions = io.load_conditions()
    rows = []
    for variant, dim in VARIANTS.items():
        for seed in SEEDS:
            run_id = f"{variant}_s{seed}"
            rng = np.random.default_rng(hash(run_id) % 2**32)
            (out_root / run_id).mkdir(parents=True, exist_ok=True)
            for c in conditions.itertuples():
                split = io.load_split(c.split_file)
                out = fake(split, c.condition_id, dim, rng)
                np.savez(out_root / run_id / f"{c.condition_id}.npz", **out)
                rows.append({"run_id": run_id, "model": variant, "seed": seed,
                             "condition_id": c.condition_id,
                             "path": f"outputs/{run_id}/{c.condition_id}.npz",
                             "n_images": len(split),
                             "accuracy": float((out["logits"].argmax(1) == out["y_true"]).mean())})
    pd.DataFrame(rows).to_csv(out_root / "manifest.csv", index=False)
    print(f"Wrote {len(rows)} fake output files to {out_root}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
