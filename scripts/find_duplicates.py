"""Find duplicate images filed under different patients, across all splits.

africa_test turned out to hold 47 pixel-identical pairs under different patient IDs.
This checks every pair of images in spain_train, spain_val, spain_test and
africa_test, within and across splits, Spain against Africa included:

- exact: identical pixels (same size, same grey values),
- near: the same picture after resizing, re-encoding or small edits. Candidates come
  from a perceptual hash (DCT of a 32x32 thumbnail, 64 bits) within MAX_HAMMING bits,
  and are kept only if the two 64x64 thumbnails correlate at MIN_CORRELATION or more
  over the pixels that are not background in either image. The hash alone isn't enough:
  the black background and fan outline make different scans from one machine look alike.

Pairs from the same patient are skipped, since frames of one scan are expected to look
alike. Matches are grouped (if A matches B and B matches C, all three are one group),
and so are the patients they connect. In each group the image from the earliest split
(spain_train, spain_val, spain_test, africa_test), then the lowest patient and image ID,
is kept as the original; every image from another patient is listed as a duplicate.

Writes one row per duplicate:

    image_id, split, duplicate_of, match_type, merge_into_patient_id

merge_into_patient_id is the patient the duplicate's patient should be merged into: the
first patient, in the same order, of all patients connected by matches.

Next to the CSV it also saves duplicate_pairs.csv (every matched pair, plus "borderline"
pairs at correlation 0.85-0.95, which are not counted) and near_duplicates.png (near and
borderline pairs side by side). On africa_test the borderline band holds both other frames
of an already-duplicated scan and different scans that merely look alike, so it is left
for checking by eye; patients linked by such a frame are usually merged anyway through
their other views.
Run in Colab after the template's setup cells:

    !python scripts/find_duplicates.py
"""
import argparse
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import DRIVE_ROOT, data_path  # noqa: E402

SPLITS = ["spain_train", "spain_val", "spain_test", "africa_test"]
MAX_HAMMING = 10        # of 64 hash bits; candidates for a near match
MIN_CORRELATION = 0.95  # thumbnail correlation needed to call a candidate a near match
BORDERLINE = 0.85       # pairs from here up to MIN_CORRELATION are listed, not counted
BACKGROUND = 10         # grey level; pixels at or below it in both images are ignored
THUMB = 64              # thumbnail side for the correlation check
PAIRS_PER_CHUNK = 5000  # candidate pairs checked at once (memory)


def fingerprint(gray: np.ndarray) -> tuple[str, np.uint64, np.ndarray]:
    """Exact pixel hash, 64-bit perceptual hash and a 64x64 thumbnail of a greyscale image."""
    exact = hashlib.md5(f"{gray.shape}".encode() + gray.tobytes()).hexdigest()
    image = Image.fromarray(gray)
    small = np.asarray(image.resize((32, 32), Image.LANCZOS), dtype=np.float32)
    low = dctn(small, norm="ortho")[:8, :8].ravel()
    bits = low > np.median(low[1:])                          # DC term left out of the median
    phash = np.packbits(bits).view(">u8")[0].astype(np.uint64)
    thumb = np.asarray(image.resize((THUMB, THUMB), Image.LANCZOS), dtype=np.float32)
    return exact, phash, thumb


def fingerprint_file(path: Path) -> tuple[str, np.uint64, np.ndarray]:
    with Image.open(path) as image:
        return fingerprint(np.asarray(image.convert("L")))


POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hash_candidates(hashes: np.ndarray, rows: int = 500) -> np.ndarray:
    """All pairs (i < j) whose perceptual hashes differ in at most MAX_HAMMING bits."""
    pairs = []
    for start in range(0, len(hashes), rows):
        block = hashes[start:start + rows, None] ^ hashes[None, :]
        distance = POPCOUNT[block.view(np.uint8)].reshape(*block.shape, 8).sum(-1)
        i, j = np.nonzero(distance <= MAX_HAMMING)
        i = i + start
        keep = i < j
        pairs.append(np.stack([i[keep], j[keep]], axis=1))
    return np.concatenate(pairs) if pairs else np.empty((0, 2), dtype=int)


def correlations(thumbs: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Correlation of each pair's thumbnails over pixels that are foreground in either one."""
    out = np.empty(len(pairs), dtype=np.float32)
    for start in range(0, len(pairs), PAIRS_PER_CHUNK):
        chunk = pairs[start:start + PAIRS_PER_CHUNK]
        a = thumbs[chunk[:, 0]].reshape(len(chunk), -1)
        b = thumbs[chunk[:, 1]].reshape(len(chunk), -1)
        used = (a > BACKGROUND) | (b > BACKGROUND)
        n = used.sum(1).clip(min=1)
        a_mean = (a * used).sum(1) / n
        b_mean = (b * used).sum(1) / n
        da = (a - a_mean[:, None]) * used
        db = (b - b_mean[:, None]) * used
        denominator = np.sqrt((da ** 2).sum(1) * (db ** 2).sum(1))
        out[start:start + len(chunk)] = np.where(denominator > 0, (da * db).sum(1) / np.maximum(denominator, 1e-6), 0)
    return out


class Groups:
    """Union-find over hashable items."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def find_duplicates(df: pd.DataFrame, exact: list[str], hashes: np.ndarray,
                    thumbs: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-patient matches between the images of df (one row per image, in the same
    order as the fingerprints). Returns the duplicates table and every matched pair."""
    patient = df["patient_id"].to_numpy()
    exact = np.asarray(exact)

    # Exact: every cross-patient pair within a group of identical pixel hashes
    edges = []
    for members in pd.Series(np.arange(len(df))).groupby(exact).groups.values():
        members = list(members)
        for x, i in enumerate(members):
            for j in members[x + 1:]:
                if patient[i] != patient[j]:
                    edges.append((i, j, "exact", 1.0))

    # Near: hash candidates from different patients, confirmed by thumbnail correlation
    candidates = hash_candidates(hashes)
    candidates = candidates[(patient[candidates[:, 0]] != patient[candidates[:, 1]])
                            & (exact[candidates[:, 0]] != exact[candidates[:, 1]])]
    scores = correlations(thumbs, candidates)
    print(f"near: {len(candidates)} cross-patient hash candidates, "
          f"{(scores >= MIN_CORRELATION).sum()} at correlation >= {MIN_CORRELATION}")
    for (i, j), r in zip(candidates[scores >= MIN_CORRELATION], scores[scores >= MIN_CORRELATION]):
        edges.append((int(i), int(j), "near", float(r)))
    borderline = (scores >= BORDERLINE) & (scores < MIN_CORRELATION)
    unconfirmed = [(int(i), int(j), "borderline", float(r))
                   for (i, j), r in zip(candidates[borderline], scores[borderline])]

    order = df["split"].map({s: k for k, s in enumerate(SPLITS)}).to_numpy()
    image_key = lambda i: (order[i], patient[i], df["image_id"].iat[i])  # noqa: E731
    patient_split = pd.Series(order, index=patient).groupby(level=0).min()
    patient_key = lambda p: (patient_split[p], p)  # noqa: E731

    images, patients = Groups(), Groups()
    for i, j, _, _ in edges:
        images.union(i, j)
        patients.union(patient[i], patient[j])

    components = {}
    for i in images.parent:
        components.setdefault(images.find(i), []).append(i)
    first_patient = {}
    for p in patients.parent:
        root = patients.find(p)
        if root not in first_patient or patient_key(p) < patient_key(first_patient[root]):
            first_patient[root] = p

    rows = []
    for members in components.values():
        original = min(members, key=image_key)
        for i in sorted(members, key=image_key):
            if patient[i] == patient[original]:
                continue
            rows.append({
                "image_id": df["image_id"].iat[i],
                "split": df["split"].iat[i],
                "duplicate_of": df["image_id"].iat[original],
                "match_type": "exact" if exact[i] == exact[original] else "near",
                "merge_into_patient_id": first_patient[patients.find(patient[i])],
            })
    duplicates = pd.DataFrame(rows, columns=["image_id", "split", "duplicate_of", "match_type",
                                             "merge_into_patient_id"]
                              ).sort_values(["merge_into_patient_id", "image_id"], ignore_index=True)
    matched = pd.DataFrame([{"a": df["image_id"].iat[i], "b": df["image_id"].iat[j], "match_type": t,
                             "correlation": round(r, 4)} for i, j, t, r in edges + unconfirmed],
                           columns=["a", "b", "match_type", "correlation"])
    return duplicates, matched


def near_sheet(df: pd.DataFrame, matched: pd.DataFrame, target: Path, n: int = 40, width: int = 240) -> None:
    """Near matches (lowest correlation first), then borderline pairs below the threshold
    (highest first), side by side, to check the threshold by eye."""
    near = pd.concat([matched[matched.match_type == "near"].sort_values("correlation").head(n // 2),
                      matched[matched.match_type == "borderline"].sort_values("correlation", ascending=False)
                      .head(n // 2)])
    if near.empty:
        return
    path = df.set_index("image_id")["path"]
    height = width * 3 // 4 + 20
    sheet = Image.new("L", (2 * width + 10, height * len(near)), 255)
    draw = ImageDraw.Draw(sheet)
    for k, row in enumerate(near.itertuples()):
        for col, image_id in enumerate((row.a, row.b)):
            image = Image.open(data_path(path[image_id])).convert("L")
            image.thumbnail((width, width * 3 // 4))
            sheet.paste(image, (col * (width + 10), k * height + 18))
        draw.text((3, k * height + 3), f"{row.match_type}: {row.a} | {row.b} | r={row.correlation}", fill=0)
    sheet.save(target)
    print(f"saved {target}")


def summarise(duplicates: pd.DataFrame, df: pd.DataFrame) -> None:
    if duplicates.empty:
        print("no cross-patient duplicates")
        return
    info = df.set_index("image_id")
    d = duplicates.assign(original_split=duplicates.duplicate_of.map(info["split"]),
                          same_label=duplicates.image_id.map(info["label"]).values
                          == duplicates.duplicate_of.map(info["label"]).values)
    print(f"\n{len(d)} duplicate images "
          f"({(d.match_type == 'exact').sum()} exact, {(d.match_type == 'near').sum()} near)")
    print(d.groupby(["original_split", "split", "match_type"]).size().rename("images").to_string())
    if (~d.same_label).any():
        print(f"\n{(~d.same_label).sum()} duplicates carry a different label from their original:")
        print(d.loc[~d.same_label, ["image_id", "duplicate_of", "match_type"]].to_string(index=False))
    patient_split = info.groupby("patient_id")["split"].first()
    moved = d.drop_duplicates("merge_into_patient_id")
    groups = d.groupby("merge_into_patient_id").image_id.apply(
        lambda ids: set(patient_split[info.loc[ids, "patient_id"]]))
    leaks = [(p, sorted(s | {patient_split[p]})) for p, s in groups.items() if s - {patient_split[p]}]
    print(f"\n{len(moved)} patient groups to merge; {len(leaks)} span more than one split")
    for p, splits in leaks[:20]:
        print(f"  {p}: {', '.join(splits)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--out", default=str(DRIVE_ROOT / "outputs" / "duplicates.csv"))
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(ROOT / "splits" / f"{s}.csv", keep_default_na=False).assign(split=s)
                    for s in args.splits], ignore_index=True)
    with ProcessPoolExecutor() as pool:
        prints = list(pool.map(fingerprint_file, [data_path(p) for p in df["path"]], chunksize=32))
    print(f"fingerprinted {len(df)} images from {', '.join(args.splits)}")
    exact = [p[0] for p in prints]
    hashes = np.array([p[1] for p in prints], dtype=np.uint64)
    thumbs = np.stack([p[2] for p in prints])

    duplicates, matched = find_duplicates(df, exact, hashes, thumbs)
    summarise(duplicates, df)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    duplicates.to_csv(out, index=False)
    matched.to_csv(out.with_name("duplicate_pairs.csv"), index=False)
    print(f"\nsaved {out} and {out.with_name('duplicate_pairs.csv')}")
    near_sheet(df, matched, out.with_name("near_duplicates.png"))


if __name__ == "__main__":
    main()
