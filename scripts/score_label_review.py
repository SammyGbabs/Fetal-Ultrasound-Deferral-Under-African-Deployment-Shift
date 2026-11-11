"""Score a returned blind label review.

Reads eval/label_review/answer_sheet.csv (filled in by the reviewer, with
'view' as abdomen, brain, femur, thorax, other or unsure) and the private
key.csv, and reports, separately for controls and candidates, how often the
reviewer agrees with the dataset label and with the models.

    !python scripts/score_label_review.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import DRIVE_ROOT  # noqa: E402

d = Path(sys.argv[1]) if len(sys.argv) > 1 else DRIVE_ROOT / "eval/label_review"
key = pd.read_csv(d / "key.csv", keep_default_na=False)
ans = pd.read_csv(d / "answer_sheet.csv", keep_default_na=False)
m = key.merge(ans.rename(columns={"image": "review_name"}), on="review_name")
m["view"] = m["view"].str.strip().str.lower()
m["agrees_with_label"] = m["view"] == m["label"]
m["agrees_with_models"] = (m["view"] == m["most_predicted_instead"]) & (m["kind"] == "candidate")

for kind in ("control", "candidate"):
    k = m[m.kind == kind]
    print(f"{kind}s: {len(k)}; reviewer agrees with dataset label on {k.agrees_with_label.sum()}")
c = m[m.kind == "candidate"]
print(f"candidates where the reviewer agrees with the models instead: {c.agrees_with_models.sum()}")
print("\n" + c[["image_id", "label", "most_predicted_instead", "view", "confidence_1_to_5",
              "comment"]].to_string(index=False))
m.to_csv(d / "review_scored.csv", index=False)
print(f"\nSaved {d / 'review_scored.csv'}")
