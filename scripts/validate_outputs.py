"""T6 step 1: check every saved output file against contract D before any analysis.

Run in Colab after the template's setup cells (outputs are read from Drive):
    !python scripts/validate_outputs.py
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

n_runs = manifest.run_id.nunique()
print(f"{n_runs} runs, {manifest.variant.nunique()} variants, {len(manifest)} files in manifest")
expected = n_runs * len(conditions)
assert len(manifest) == expected, f"expected {expected} files, manifest has {len(manifest)}"

bad = 0
feature_dims = {}
for row in manifest.itertuples():
    out = io.load_output(outputs_root, row.run_id, row.condition_id)
    problems = io.check_output(out, splits[conditions.loc[row.condition_id, "split_file"]])
    feature_dims.setdefault(row.variant, out["features"].shape[1])
    if problems:
        bad += 1
        print(f"FAIL {row.run_id}/{row.condition_id}: {problems}")
print("\nFeature dimension per variant:", feature_dims)
print(f"\n{len(manifest) - bad} of {len(manifest)} files pass contract D.")
sys.exit(1 if bad else 0)
