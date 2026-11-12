"""T6 step 1: check every saved output file against contract D before any analysis.

Each manifest row is checked against its own split file, so runs that have only
some conditions (e.g. quantized runs) are fine. Runs with fewer conditions than
the full set are listed, so a missing file is never silent.

Run in Colab after the template's setup cells:
    !python scripts/validate_outputs.py /content/outputs_local
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval import io  # noqa: E402
from src.paths import DRIVE_ROOT  # noqa: E402

outputs_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DRIVE_ROOT / "outputs"
manifest = io.load_manifest(outputs_root)
conditions = io.load_conditions().set_index("condition_id")
splits = {f: io.load_split(f) for f in conditions["split_file"].unique()}

unknown = sorted(set(manifest.condition_id) - set(conditions.index))
if unknown:
    sys.exit(f"manifest lists conditions not in conditions.csv: {unknown}")
dupes = manifest.duplicated(["run_id", "condition_id"]).sum()
if dupes:
    sys.exit(f"manifest has {dupes} repeated run/condition rows")

print(f"{manifest.run_id.nunique()} runs, {manifest.variant.nunique()} variants, "
      f"{len(manifest)} files in manifest")
n_all = len(conditions)
counts = manifest.groupby("variant").condition_id.nunique()
partial = counts[counts < n_all]
if len(partial):
    print(f"\nVariants with only some of the {n_all} conditions (expected for quantized runs):")
    for variant, n in partial.items():
        print(f"  {variant}: {n} conditions ({', '.join(sorted(io.conditions_of(manifest, variant)))})")
uneven = [v for v in manifest.variant.unique()
          if manifest[manifest.variant == v].groupby("run_id").condition_id.nunique().nunique() > 1]
if uneven:
    print(f"\nWARNING: seeds of these variants have different conditions: {uneven}. "
          "Only the conditions all seeds share will be evaluated.")

bad = 0
feature_dims = {}
for row in manifest.itertuples():
    path = outputs_root / row.run_id / f"{row.condition_id}.npz"
    if not path.exists():
        bad += 1
        print(f"FAIL {row.run_id}/{row.condition_id}: file listed in manifest but missing")
        continue
    out = io.load_output(outputs_root, row.run_id, row.condition_id)
    problems = io.check_output(out, splits[conditions.loc[row.condition_id, "split_file"]])
    feature_dims.setdefault(row.variant, out["features"].shape[1])
    if problems:
        bad += 1
        print(f"FAIL {row.run_id}/{row.condition_id}: {problems}")
print("\nFeature dimension per variant:", feature_dims)
print(f"\n{len(manifest) - bad} of {len(manifest)} files pass contract D.")
sys.exit(1 if bad else 0)
