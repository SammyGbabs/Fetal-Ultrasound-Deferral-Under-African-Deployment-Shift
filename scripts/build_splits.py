"""T2: build the four frozen split CSVs (contract A).

Reads the original metadata from the extracted datasets, keeps the 4 fetal
planes, carves a patient-level validation set out of the official Spanish
training split, and writes:

    splits/spain_train.csv  splits/spain_val.csv
    splits/spain_test.csv   splits/africa_test.csv

Run once, in Colab, after the template's setup cells:
    !python scripts/build_splits.py
Then commit the four CSVs to Git. They must never be regenerated afterwards,
so everyone trains and tests on exactly the same images.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import data_path  # noqa: E402

LABELS = {"Fetal abdomen": 0, "Fetal brain": 1, "Fetal femur": 2, "Fetal thorax": 3}
VAL_FRACTION = 0.15
SEED = 42

SPAIN_META = "data/raw/spain/FETAL_PLANES_DB_data.xlsx"
SPAIN_IMAGES = "data/raw/spain/Images"
AFRICA_META = "data/raw/africa/Zenodo_dataset/African_planes_database.csv"
AFRICA_IMAGES = "data/raw/africa/Zenodo_dataset"

COLUMNS = ["image_id", "path", "label", "label_name", "patient_id", "source",
           "machine", "operator", "country", "official_split"]


def build_spain(meta_file: Path) -> pd.DataFrame:
    df = pd.read_excel(meta_file)
    df.columns = [c.strip() for c in df.columns]          # "Train " -> "Train"
    df["Plane"] = df["Plane"].str.strip()
    df = df[df["Plane"].isin(LABELS)].copy()
    df["image_id"] = df["Image_name"].str.replace(".png", "", regex=False)
    return pd.DataFrame({
        "image_id": df["image_id"],
        "path": SPAIN_IMAGES + "/" + df["image_id"] + ".png",
        "label": df["Plane"].map(LABELS),
        "label_name": df["Plane"],
        "patient_id": "ES_" + df["Patient_num"].astype(int).map("{:05d}".format),
        "source": "spain",
        "machine": df["US_Machine"].str.strip(),
        "operator": df["Operator"].astype(str).str.strip(),
        "country": "Spain",
        "official_split": np.where(df["Train"] == 1, "train", "test"),
    })


def build_africa(meta_file: Path) -> pd.DataFrame:
    df = pd.read_csv(meta_file)
    df.columns = [c.strip() for c in df.columns]
    df["Plane"] = df["Plane"].str.strip()
    code = df["Filename"].str.extract(r"_([A-Z]{3})_")[0]   # DZA, EGY, GHA, MWI, UGA
    return pd.DataFrame({
        "image_id": df["Filename"],
        "path": AFRICA_IMAGES + "/" + df["Center"] + "/" + df["Filename"] + ".png",
        "label": df["Plane"].map(LABELS),
        "label_name": df["Plane"],
        "patient_id": code + "_" + df["Patient_num"].astype(int).map("{:04d}".format),
        "source": "africa",
        "machine": "",
        "operator": "",
        "country": df["Center"],
        "official_split": np.where(df["Train"] == 1, "train", "test"),
    })


def split_validation(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out ~15% of training patients, stratified by each patient's main machine.

    Voluson S10 patients stay in training: S10 is the 'unseen device' test,
    and the 11 training images are kept exactly as in the official split.
    """
    rng = np.random.default_rng(SEED)
    main_machine = train.groupby("patient_id")["machine"].agg(lambda m: m.mode().iloc[0])
    val_patients = []
    for machine, patients in main_machine.groupby(main_machine):
        if machine == "Voluson S10":
            continue
        ids = np.array(sorted(patients.index))
        n_val = int(round(len(ids) * VAL_FRACTION))
        val_patients.extend(rng.choice(ids, size=n_val, replace=False))
    is_val = train["patient_id"].isin(val_patients)
    return train[~is_val], train[is_val]


def check(splits: dict, check_files: bool) -> None:
    sp = pd.concat([splits[k] for k in ("spain_train", "spain_val", "spain_test")])
    assert len(sp) == 6561, f"expected 6,561 Spanish images, got {len(sp)}"
    assert len(splits["africa_test"]) == 450, "expected 450 African images"
    assert splits["africa_test"]["patient_id"].nunique() == 127, "expected 127 African patients"
    for a in ("spain_train", "spain_val", "spain_test"):
        for b in ("spain_train", "spain_val", "spain_test"):
            if a < b:
                shared = set(splits[a].patient_id) & set(splits[b].patient_id)
                assert not shared, f"patients in both {a} and {b}: {sorted(shared)[:5]}"
    for name, df in splits.items():
        assert df["label"].notna().all(), f"unmapped plane name in {name}"
        assert df["image_id"].is_unique, f"duplicate image_id in {name}"
        if check_files:
            missing = [p for p in df["path"] if not data_path(p).exists()]
            assert not missing, f"{len(missing)} missing files in {name}, e.g. {missing[:3]}"
    print("All checks passed.\n")


def report(splits: dict) -> None:
    for name, df in splits.items():
        print(f"{name}: {len(df)} images, {df.patient_id.nunique()} patients")
        print(df.groupby("label_name").size().to_string(), "\n")
    print("Spanish test images by machine:")
    print(splits["spain_test"].groupby("machine").size().to_string(), "\n")
    print("African images by country and plane:")
    print(pd.crosstab(splits["africa_test"].country, splits["africa_test"].label_name).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-file-check", action="store_true",
                        help="only for testing without images present")
    parser.add_argument("--spain-meta", default=None)
    parser.add_argument("--africa-meta", default=None)
    args = parser.parse_args()

    spain = build_spain(Path(args.spain_meta) if args.spain_meta else data_path(SPAIN_META))
    africa = build_africa(Path(args.africa_meta) if args.africa_meta else data_path(AFRICA_META))

    train, val = split_validation(spain[spain.official_split == "train"])
    splits = {
        "spain_train": train,
        "spain_val": val,
        "spain_test": spain[spain.official_split == "test"],
        "africa_test": africa,
    }
    check(splits, check_files=not args.skip_file_check)
    report(splits)

    out = ROOT / "splits"
    out.mkdir(exist_ok=True)
    for name, df in splits.items():
        df[COLUMNS].reset_index(drop=True).to_csv(out / f"{name}.csv", index=False)
    print(f"\nWrote 4 CSVs to {out}. Commit them to Git.")


if __name__ == "__main__":
    main()
