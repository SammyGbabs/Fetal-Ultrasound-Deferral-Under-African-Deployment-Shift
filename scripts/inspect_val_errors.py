"""T7 check: which Spanish validation images are wrong in many runs?

An image that nearly every model gets wrong may be mislabelled. Prints the
most-missed images and saves them to eval/figures/val_errors/ for viewing.

    !python scripts/inspect_val_errors.py /content/outputs_local
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval import io  # noqa: E402
from src.paths import DRIVE_ROOT, data_path  # noqa: E402

outputs_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DRIVE_ROOT / "outputs"
manifest = io.load_manifest(outputs_root)
split = io.load_split("spain_val.csv").set_index("image_id")
labels = {0: "abdomen", 1: "brain", 2: "femur", 3: "thorax"}

wrong = []
for run_id in sorted(manifest.run_id.unique()):
    p = pd.read_csv(outputs_root / run_id / "spain_val_preds.csv")
    for r in p[p.y_pred != p.y_true].itertuples():
        wrong.append({"run_id": run_id, "variant": io.variant_of(run_id),
                      "image_id": str(r.image_id), "y_pred": labels[r.y_pred]})
wrong = pd.DataFrame(wrong)
n_runs = manifest.run_id.nunique()
summary = (wrong.groupby("image_id")
           .agg(runs_wrong=("run_id", "nunique"), variants=("variant", lambda v: sorted(set(v))),
                predicted_as=("y_pred", lambda v: v.value_counts().to_dict()))
           .sort_values("runs_wrong", ascending=False))
summary["label"] = [split.loc[i, "label_name"] for i in summary.index]
summary["patient"] = [split.loc[i, "patient_id"] for i in summary.index]
print(f"{len(summary)} validation images are wrong in at least one of {n_runs} runs.\n")
print(summary.head(10).to_string())

out = DRIVE_ROOT / "eval/figures/val_errors"
out.mkdir(parents=True, exist_ok=True)
for image_id in summary.head(5).index:
    src = data_path(split.loc[image_id, "path"])
    if src.exists():
        shutil.copy(src, out / f"{summary.loc[image_id, 'runs_wrong']}runs_{image_id}.png")
    else:
        print(f"image not found (run template cell 6 first?): {src}")
print(f"\nTop 5 copied to {out} for viewing.")
