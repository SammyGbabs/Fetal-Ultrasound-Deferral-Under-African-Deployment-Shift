"""Find possible label errors in the Spanish data and prepare a blind review set.

Step 1. For spain_test and spain_val, count in how many of the 45 runs each
image is misclassified, and which view the runs predict instead. Images most
runs disagree with are label-error candidates. Duplicate copies are skipped.

Step 2. Copy the candidates (wrong in at least --min-frac of runs, at most --max
images) plus a few control images (all runs correct and confident) into
eval/label_review/blind/ under neutral names, in random order. The reviewer
sees only these images and fills in eval/label_review/answer_sheet.csv; the key
linking names to images, labels and predictions is kept in
eval/label_review/key.csv and must NOT be shared with the reviewer.

    !python scripts/find_label_candidates.py --outputs /content/outputs_local

Images are read from the local copy if present, otherwise from Drive.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval import io  # noqa: E402
from src.paths import DRIVE_ROOT, data_path  # noqa: E402

LABELS = {0: "abdomen", 1: "brain", 2: "femur", 3: "thorax"}

parser = argparse.ArgumentParser()
parser.add_argument("--outputs", default=str(DRIVE_ROOT / "outputs"))
parser.add_argument("--eval-dir", default=str(DRIVE_ROOT / "eval"))
parser.add_argument("--splits", nargs="+", default=["spain_test", "spain_val"])
parser.add_argument("--min-frac", type=float, default=0.5,
                    help="review images misclassified by at least this share of runs")
parser.add_argument("--max", type=int, default=20, help="at most this many candidates")
parser.add_argument("--controls", type=int, default=5, help="confidently correct images mixed in")
parser.add_argument("--seed", type=int, default=2026)
args = parser.parse_args()
outputs_root = Path(args.outputs)
review_dir = Path(args.eval_dir) / "label_review"
duplicates = io.load_duplicates()
run_ids = sorted(io.load_manifest(outputs_root).run_id.unique())
n_runs = len(run_ids)

rows = []
for split_name in args.splits:
    split = io.load_split(f"{split_name}.csv")
    keep, _ = io.evaluation_rows(f"{split_name}.csv", split, duplicates)
    preds, probs = [], []
    for r in run_ids:
        with np.load(outputs_root / r / f"{split_name}.npz", allow_pickle=True) as z:
            logits = z["logits"].astype(np.float64)
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        preds.append(p.argmax(1))
        probs.append(p)
    preds, probs = np.stack(preds), np.stack(probs)          # (runs, N), (runs, N, 4)
    y = split["label"].to_numpy()
    wrong = preds != y
    for i in keep:
        alt = np.bincount(preds[wrong[:, i], i], minlength=4) if wrong[:, i].any() else np.zeros(4, int)
        rows.append({
            "split": split_name, "image_id": split.image_id[i], "path": split.path[i],
            "patient_id": split.patient_id[i], "label": LABELS[y[i]],
            "runs_wrong": int(wrong[:, i].sum()), "frac_wrong": wrong[:, i].mean(),
            "most_predicted_instead": LABELS[int(alt.argmax())] if alt.any() else "",
            "mean_prob_of_label": float(probs[:, i, y[i]].mean()),
            "variants_wrong": len({io.variant_of(r) for r, w in zip(run_ids, wrong[:, i]) if w}),
        })

table = pd.DataFrame(rows).sort_values(["runs_wrong", "mean_prob_of_label"], ascending=[False, True])
review_dir.mkdir(parents=True, exist_ok=True)
table[table.runs_wrong > 0].to_csv(review_dir / "all_disagreements.csv", index=False)

print(f"{len(table)} images checked across {n_runs} runs.")
for lo, hi, name in ((0.5, 1.01, "at least half"), (0.25, 0.5, "a quarter to half"), (0.0001, 0.25, "under a quarter")):
    n = ((table.frac_wrong >= lo) & (table.frac_wrong < hi)).sum()
    print(f"  wrong in {name} of runs: {n}")
cands = table[table.frac_wrong >= args.min_frac].head(args.max)
print(f"\nCandidates for review (wrong in >= {args.min_frac:.0%} of runs): {len(cands)}")
print(cands[["split", "image_id", "label", "most_predicted_instead", "runs_wrong",
             "variants_wrong"]].to_string(index=False))

# controls: correct in every run (fewest errors if none are), most confident, spread over the views
per_view = max(1, -(-args.controls // 4))
pool = table[~table.image_id.isin(cands.image_id)]
easy = pool[pool.runs_wrong == pool.runs_wrong.min()].sort_values("mean_prob_of_label", ascending=False)
picked = [easy[easy.label == v].head(50) for v in LABELS.values()]
picked = [g.sample(min(len(g), per_view), random_state=args.seed) for g in picked if len(g)]
controls = pd.concat(picked).sample(frac=1, random_state=args.seed).head(args.controls)
review = pd.concat([cands.assign(kind="candidate"), controls.assign(kind="control")])
review = review.sample(frac=1, random_state=args.seed).reset_index(drop=True)
review["review_name"] = [f"image_{i + 1:02d}.png" for i in range(len(review))]

blind = review_dir / "blind"
if blind.exists():
    shutil.rmtree(blind)
blind.mkdir(parents=True)
copied = 0
for r in review.itertuples():
    src = data_path(r.path)
    if src.exists():
        shutil.copy(src, blind / r.review_name)
        copied += 1
    else:
        print(f"image not found: {src}")
review.to_csv(review_dir / "key.csv", index=False)
pd.DataFrame({"image": review.review_name, "view": "", "confidence_1_to_5": "", "comment": ""}
             ).to_csv(review_dir / "answer_sheet.csv", index=False)
print(f"\nCopied {copied} of {len(review)} images ({len(cands)} candidates, {len(controls)} controls) to {blind}")
print("Share ONLY the 'blind' folder and answer_sheet.csv with the reviewer. Keep key.csv private.")
