"""Checks for the T6/T7 evaluation code. Run:  python -m pytest tests/ -q"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval import calibration, deferral, io, metrics, scores  # noqa: E402

rng = np.random.default_rng(0)


def test_temperature_recovers_known_value():
    # labels sampled from softmax(z / 2): the best temperature for logits z is about 2
    z = rng.normal(0, 3, (20000, 4))
    p = calibration.probs(z, 2.0)
    y = np.array([rng.choice(4, p=row) for row in p])
    assert abs(calibration.fit_temperature(z, y) - 2.0) < 0.15


def test_ece_zero_for_perfect_calibration():
    p = calibration.probs(rng.normal(0, 2, (50000, 4)))
    y = np.array([rng.choice(4, p=row) for row in p])
    assert metrics.ece(p, y) < 0.02


def test_ece_large_for_overconfidence():
    p = np.tile([0.97, 0.01, 0.01, 0.01], (1000, 1))
    y = np.r_[np.zeros(500, int), np.ones(500, int)]          # only 50% right
    assert metrics.ece(p, y) > 0.4


def test_balanced_accuracy_uses_present_classes():
    y = np.array([0, 0, 1, 1, 2, 2])                          # no thorax, like Ghana/Uganda
    pred = np.array([0, 0, 1, 3, 2, 2])
    assert np.isclose(metrics.balanced_accuracy(y, pred), (1 + 0.5 + 1) / 3)


def test_perfect_uncertainty_gives_best_risk_coverage():
    y = np.zeros(100, int)
    pred = np.r_[np.zeros(80, int), np.ones(20, int)]
    u_good = np.r_[np.zeros(80), np.ones(20)]                 # errors are the most uncertain
    u_bad = -u_good
    assert deferral.accuracy_at_coverage(y, pred, u_good, 0.8) == 1.0
    assert metrics.aurc(y, pred, u_good) < metrics.aurc(y, pred, u_bad)


def test_deployment_threshold_hits_target_on_same_data():
    u = rng.random(1000)
    thr = deferral.threshold_for_coverage(u, 0.8)
    d = deferral.apply(np.zeros(1000, int), np.zeros(1000, int), u, thr)
    assert abs(d["coverage"] - 0.8) < 0.01


def test_mahalanobis_flags_shifted_features():
    centres = rng.normal(0, 3, (4, 16))
    y = rng.integers(0, 4, 2000)
    train = centres[y] + rng.normal(0, 1, (2000, 16))
    maha = scores.Mahalanobis().fit(train, y)
    familiar = centres[y[:500]] + rng.normal(0, 1, (500, 16))
    shifted = centres[y[:500]] + rng.normal(2, 1, (500, 16))
    assert metrics.ood_auroc(maha.score(familiar), maha.score(shifted)) > 0.9


def test_mc_mutual_information_nonnegative_and_zero_without_disagreement():
    same = np.repeat(rng.normal(0, 2, (50, 1, 4)), 20, axis=1)
    _, _, mi = scores.mc_dropout(same)
    assert np.allclose(mi, 0, atol=1e-9)
    _, _, mi = scores.mc_dropout(rng.normal(0, 2, (50, 20, 4)))
    assert (mi >= -1e-9).all()


# ---- numerical-stability fix -------------------------------------------------

def test_log_scores_keep_the_same_ranking_as_plain_scores():
    """log(1 - max p) and log(entropy) must order images exactly like 1 - max p and entropy."""
    z = rng.normal(0, 2, (5000, 4))
    p = softmax(z, axis=1)
    assert spearmanr(scores.msp(z)[1], 1 - p.max(1)).statistic > 0.999999
    assert spearmanr(scores.predictive_entropy(z)[1], scores.entropy(p)).statistic > 0.999999
    _, u_ens, _ = scores.ensemble([z, z + rng.normal(0, 0.5, z.shape)])
    p_ens = (p + softmax(z + 0, axis=1)) / 2
    assert np.isfinite(u_ens).all()


def test_scores_have_no_rounding_ties_at_extreme_temperature():
    """At T = 0.05, plain 1 - max p collapses to 0 for most images; the log form must not."""
    margins = np.linspace(2, 30, 500)                         # all different confidences
    z = np.zeros((500, 4))
    z[:, 0] = margins
    old = 1 - softmax(z / 0.05, axis=1).max(1)                # the previous formula
    new = scores.msp(z, T=0.05)[1]
    assert (old == 0).sum() > 100                             # the bug: hundreds of exact ties
    assert len(np.unique(new)) == 500                         # fixed: every image distinct
    assert (np.diff(new) < 0).all()                           # bigger margin = less uncertain
    new_h = scores.predictive_entropy(z, T=0.05)[1]
    assert len(np.unique(new_h)) == 500 and (np.diff(new_h) < 0).all()
    _, ens_u, _ = scores.ensemble([z, z], temps=[0.05, 0.05])
    assert len(np.unique(ens_u)) == 500


def test_outputs_load_in_double_precision(tmp_path):
    out = {"image_id": np.array(["a", "b"]), "y_true": np.array([0, 1]),
           "logits": np.zeros((2, 4), np.float32), "features": np.zeros((2, 3), np.float32),
           "mc_logits": np.zeros((2, 20, 4), np.float32)}
    (tmp_path / "run_s0").mkdir()
    np.savez(tmp_path / "run_s0" / "cond.npz", **out)
    loaded = io.load_output(tmp_path, "run_s0", "cond")
    assert all(loaded[k].dtype == np.float64 for k in ("logits", "features", "mc_logits"))


# ---- end to end --------------------------------------------------------------

def test_end_to_end_on_fake_outputs(tmp_path):
    fake = tmp_path / "outputs"
    subprocess.run([sys.executable, ROOT / "scripts/make_fake_outputs.py", fake], check=True)
    subprocess.run([sys.executable, ROOT / "scripts/validate_outputs.py", fake], check=True)
    eval_dir = tmp_path / "eval"
    subprocess.run([sys.executable, ROOT / "scripts/run_eval.py",
                    "--outputs", fake, "--eval-dir", eval_dir], check=True)
    df = pd.read_csv(eval_dir / "per_run_metrics.csv")
    # 2 variants x 5 seeds x test conditions x 8 scores x 2 calibrations
    n_test = sum(not io.is_validation(c) for c in io.load_conditions().condition_id)
    assert len(df) == 2 * 5 * n_test * 8 * 2
    clean = df[(df.score == "msp") & (df.calib == "raw")]
    spain = clean[clean.condition == "spain_test"].accuracy.mean()
    africa = clean[clean.condition == "africa_test"].accuracy.mean()
    africa_q10 = clean[clean.condition == "africa_test_q10"].accuracy.mean()
    assert spain > africa > africa_q10                        # the fake shifts are detected
    maha = df[(df.score == "mahalanobis") & (df.condition == "africa_test")].ood_auroc.mean()
    assert maha > 0.9                                         # fake African features are shifted


def test_t7_matches_t6_on_fake_outputs(tmp_path):
    """T7's point estimates must equal T6's seed-averaged values exactly."""
    fake, ev = tmp_path / "outputs", tmp_path / "eval"
    subprocess.run([sys.executable, ROOT / "scripts/make_fake_outputs.py", fake], check=True)
    subprocess.run([sys.executable, ROOT / "scripts/run_eval.py",
                    "--outputs", fake, "--eval-dir", ev], check=True)
    subprocess.run([sys.executable, ROOT / "scripts/run_t7.py",
                    "--outputs", fake, "--eval-dir", ev, "--B", "20"], check=True)
    t6 = pd.read_csv(ev / "per_run_metrics.csv")
    t7 = pd.read_csv(ev / "results_main.csv")
    for method, score in (("msp", "msp"), ("ensemble", "ens_entropy")):
        for calib in ("raw", "ts"):
            for cond in ("spain_test", "africa_test", "africa_test_q10"):
                a = t6[(t6.score == score) & (t6.calib == calib) & (t6.condition == cond)]
                a = a.groupby("variant").mean(numeric_only=True)
                for metric in ("balanced_accuracy", "ece", "silent_failure", "acc_at_80",
                               "deployed_coverage_80", "errors_passed_80"):
                    b = t7[(t7.method == method) & (t7.calib == calib) &
                           (t7.group == f"{cond}_all") & (t7.metric == metric)]
                    b = b.set_index("variant").estimate
                    assert np.allclose(a[metric], b, atol=1e-6), (method, calib, cond, metric)
    # every finite estimate has a finite, ordered interval
    ok = t7[np.isfinite(t7.estimate)]
    assert np.isfinite(ok.ci_low).all() and (ok.ci_low <= ok.ci_high).all()


def test_t6_and_t7_agree_at_extreme_temperature(tmp_path):
    """The original bug: with near-certain models, T6 and T7 disagreed on deferral.

    Fake outputs with very confident logits push the fitted temperature to its
    lower bound; T6 (via pipeline) and T7 must still give identical deferral numbers.
    """
    fake, ev = tmp_path / "outputs", tmp_path / "eval"
    subprocess.run([sys.executable, ROOT / "scripts/make_fake_outputs.py", fake], check=True)
    for run_dir in fake.iterdir():                            # make every model near-certain
        if not run_dir.is_dir():
            continue
        for f in run_dir.glob("*.npz"):
            with np.load(f, allow_pickle=True) as z:
                d = {k: z[k] for k in z.files}
            y = d["y_true"]
            logits = np.full((len(y), 4), -10.0, np.float32)
            logits[np.arange(len(y)), y] = 10.0 + rng.random(len(y)).astype(np.float32)
            wrong = rng.random(len(y)) < (0.08 if "africa" in f.name else 0.0)
            logits[wrong] = logits[wrong][:, ::-1]               # a few confident errors in Africa
            d["logits"] = logits
            d["mc_logits"] = np.repeat(logits[:, None, :], 20, axis=1)
            np.savez(f, **d)
    subprocess.run([sys.executable, ROOT / "scripts/run_eval.py",
                    "--outputs", fake, "--eval-dir", ev], check=True)
    subprocess.run([sys.executable, ROOT / "scripts/run_t7.py",
                    "--outputs", fake, "--eval-dir", ev, "--B", "5"], check=True)
    temps = pd.read_csv(ev / "temperatures.csv")
    temps = temps[temps.val_condition == "spain_val"].set_index("run_id").temperature
    assert (temps < 0.5).all()                                # strong sharpening, as for EfficientNet-B0
    # confirm the old formula would have produced rounding ties on this data
    with np.load(fake / "fakecnn_s0" / "africa_test.npz", allow_pickle=True) as z:
        old = 1 - softmax(z["logits"].astype(np.float64) / temps["fakecnn_s0"], axis=1).max(1)
    assert (old == 0).mean() > 0.5
    t6 = pd.read_csv(ev / "per_run_metrics.csv")
    t7 = pd.read_csv(ev / "results_main.csv")
    a = t6[(t6.score == "msp") & (t6.calib == "ts") & (t6.condition == "africa_test")]
    a = a.groupby("variant").mean(numeric_only=True)
    for metric in ("deployed_coverage_80", "errors_passed_80"):
        b = t7[(t7.method == "msp") & (t7.calib == "ts") & (t7.group == "africa_test_all") &
               (t7.metric == metric)].set_index("variant").estimate
        assert np.allclose(a[metric], b, atol=1e-9), metric


# ---- duplicates, leaked validation images, masked thresholds ----------------

def test_evaluation_rows_drops_copies_and_merges_patients():
    split = pd.DataFrame({"image_id": ["a1", "a2", "b1", "b2", "c1"],
                          "patient_id": ["A", "A", "B", "B", "C"]})
    dup = pd.DataFrame({"image_id": ["b1", "c1"], "split": ["africa_test", "africa_test"],
                        "duplicate_of": ["a1", "b2"], "match_type": ["exact", "near"],
                        "merge_into_patient_id": ["A", "B"]})
    keep, patients = io.evaluation_rows("africa_test.csv", split, dup)
    assert list(keep) == [0, 1, 3]                            # copies b1 and c1 dropped
    assert list(patients) == ["A", "A", "A", "A", "A"]        # B joins A; C joins B, hence A
    keep, patients = io.evaluation_rows("spain_test.csv", split, dup)
    assert list(keep) == [0, 1, 2, 3, 4]                      # other splits untouched
    assert list(io.evaluation_rows("africa_test.csv", split, None)[0]) == [0, 1, 2, 3, 4]


def _fake_duplicates(path: Path, n_africa=20, n_val=5):
    """Mark some African images and some validation images as copies."""
    africa = io.load_split("africa_test.csv")
    val, train = io.load_split("spain_val.csv"), io.load_split("spain_train.csv")
    first = africa.groupby("patient_id", sort=False).head(1)  # one image per patient
    rows = []
    for i in range(n_africa):                                 # patient 2i+1's image copies patient 2i's
        copy, orig = first.iloc[2 * i + 1], first.iloc[2 * i]
        rows.append({"image_id": copy.image_id, "split": "africa_test",
                     "duplicate_of": orig.image_id, "match_type": "exact",
                     "merge_into_patient_id": orig.patient_id})
    for i in range(n_val):                                    # leaked validation images
        rows.append({"image_id": val.image_id[i], "split": "spain_val",
                     "duplicate_of": train.image_id[i], "match_type": "exact",
                     "merge_into_patient_id": train.patient_id[i]})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_deduplication_end_to_end(tmp_path):
    fake, ev, dup = tmp_path / "outputs", tmp_path / "eval", tmp_path / "duplicates.csv"
    subprocess.run([sys.executable, ROOT / "scripts/make_fake_outputs.py", fake], check=True)
    _fake_duplicates(dup, n_africa=20, n_val=5)
    for script, extra in (("run_eval.py", []), ("run_t7.py", ["--B", "5"])):
        subprocess.run([sys.executable, ROOT / "scripts" / script, "--outputs", fake,
                        "--eval-dir", ev, "--duplicates", dup, *extra], check=True)
        subprocess.run([sys.executable, ROOT / "scripts" / script, "--outputs", fake,
                        "--eval-dir", ev, "--no-dedup", *extra], check=True)
    n_africa = len(io.load_split("africa_test.csv"))

    t6 = pd.read_csv(ev / "per_run_metrics.csv")
    assert (t6[t6.condition.str.startswith("africa_test")].n == n_africa - 20).all()
    assert (t6[t6.condition == "spain_test"].n == len(io.load_split("spain_test.csv"))).all()
    masked = t6[t6.condition.str.endswith("_masked")]
    assert (masked.val_condition == "spain_val_masked").all()  # masked thresholds from masked validation
    assert (t6[~t6.condition.str.endswith("_masked")].val_condition == "spain_val").all()
    temps = pd.read_csv(ev / "temperatures.csv")
    assert (temps.n_val_images == len(io.load_split("spain_val.csv")) - 5).all()

    t7 = pd.read_csv(ev / "results_main.csv")
    a = t7[t7.group == "africa_test_all"]
    assert (a.n_images == n_africa - 20).all()
    assert (a.n_patients < io.load_split("africa_test.csv").patient_id.nunique()).all()
    free = t7[t7.group == "africa_test_caliper_free"].n_images.iloc[0]
    africa = io.load_split("africa_test.csv")
    assert free <= (~africa.country.isin(["Ghana", "Uganda"])).sum()

    # T6 and T7 still agree after deduplication
    for metric in ("balanced_accuracy", "deployed_coverage_80", "errors_passed_80"):
        x = t6[(t6.score == "msp") & (t6.calib == "raw") & (t6.condition == "africa_test")]
        x = x.groupby("variant")[metric].mean()
        y = t7[(t7.method == "msp") & (t7.calib == "raw") & (t7.group == "africa_test_all") &
               (t7.metric == metric)].set_index("variant").estimate
        assert np.allclose(x, y, atol=1e-9), metric

    # --no-dedup keeps all images and writes elsewhere
    t6_all = pd.read_csv(ev / "no_dedup" / "per_run_metrics.csv")
    assert (t6_all[t6_all.condition == "africa_test"].n == n_africa).all()
    t7_all = pd.read_csv(ev / "no_dedup" / "results_main.csv")
    assert (t7_all[t7_all.group == "africa_test_all"].n_images == n_africa).all()


def test_masking_ci_matches_t6(tmp_path):
    """Masked-versus-original point estimates must equal T6's seed means for both conditions."""
    fake, ev = tmp_path / "outputs", tmp_path / "eval"
    subprocess.run([sys.executable, ROOT / "scripts/make_fake_outputs.py", fake], check=True)
    subprocess.run([sys.executable, ROOT / "scripts/run_eval.py",
                    "--outputs", fake, "--eval-dir", ev], check=True)
    subprocess.run([sys.executable, ROOT / "scripts/run_masking_ci.py",
                    "--outputs", fake, "--eval-dir", ev, "--B", "10"], check=True)
    t6 = pd.read_csv(ev / "per_run_metrics.csv")
    m = pd.read_csv(ev / "masking_ci.csv")
    for base in ("spain_test", "africa_test"):
        for version, cond in (("original", base), ("masked", f"{base}_masked")):
            for metric in ("balanced_accuracy", "deployed_coverage_80", "errors_passed_80"):
                a = t6[(t6.score == "msp") & (t6.calib == "raw") & (t6.condition == cond)]
                a = a.groupby("variant")[metric].mean()
                b = m[(m.method == "msp") & (m.group == f"{base}_all") & (m.metric == metric)]
                assert np.allclose(a, b.set_index("variant")[version], atol=1e-9), (cond, metric)
    assert np.allclose(m.difference, m.masked - m.original)
    assert set(m.group) == {"spain_test_all", "africa_test_all", "africa_test_caliper_free"}
    ok = m[np.isfinite(m.difference)]
    assert (ok.ci_low <= ok.ci_high).all()


# ---- partial runs (e.g. quantized models with only some conditions) ----------

QUANT_CONDITIONS = ["spain_train", "spain_val", "spain_test", "africa_test", "africa_test_q10"]


def _add_partial_variant(fake: Path, source="fakecnn", name="fakecnn-int8"):
    """Copy 5 conditions of an existing variant into a new 'quantized' variant."""
    man = pd.read_csv(fake / "manifest.csv")
    new_rows = []
    for seed in range(5):
        src, dst = fake / f"{source}_s{seed}", fake / f"{name}_s{seed}"
        dst.mkdir()
        for c in QUANT_CONDITIONS:
            with np.load(src / f"{c}.npz", allow_pickle=True) as z:
                d = {k: z[k] for k in z.files}
            d["logits"] = d["logits"] + rng.normal(0, 0.05, d["logits"].shape).astype(np.float32)
            d["mc_logits"] = np.repeat(d["logits"][:, None, :], 20, axis=1)
            np.savez(dst / f"{c}.npz", **d)
            new_rows.append({"run_id": f"{name}_s{seed}", "model": name, "seed": seed,
                             "condition_id": c, "path": f"outputs/{name}_s{seed}/{c}.npz",
                             "n_images": len(d["y_true"]), "accuracy": 0.0})
    pd.concat([man, pd.DataFrame(new_rows)]).to_csv(fake / "manifest.csv", index=False)


def test_scripts_handle_partial_variants(tmp_path):
    fake, ev = tmp_path / "outputs", tmp_path / "eval"
    subprocess.run([sys.executable, ROOT / "scripts/make_fake_outputs.py", fake], check=True)
    _add_partial_variant(fake)
    v = subprocess.run([sys.executable, ROOT / "scripts/validate_outputs.py", fake],
                       capture_output=True, text=True)
    assert v.returncode == 0, v.stdout
    assert "fakecnn-int8: 5 conditions" in v.stdout

    for script, extra in (("run_eval.py", []), ("run_t7.py", ["--B", "5"]),
                          ("run_masking_ci.py", ["--B", "5"])):
        subprocess.run([sys.executable, ROOT / "scripts" / script, "--outputs", fake,
                        "--eval-dir", ev, *extra], check=True)

    t6 = pd.read_csv(ev / "per_run_metrics.csv")
    q = t6[t6.variant == "fakecnn-int8"]
    assert set(q.condition) == {"spain_test", "africa_test", "africa_test_q10"}
    assert len(q) == 5 * 3 * 8 * 2
    n_test = sum(not io.is_validation(c) for c in io.load_conditions().condition_id)
    assert len(t6[t6.variant == "fakecnn"]) == 5 * n_test * 8 * 2      # full variants unchanged

    t7 = pd.read_csv(ev / "results_main.csv")
    assert {"spain_test_all", "africa_test_all", "africa_test_q10_all"} <= \
        set(t7[t7.variant == "fakecnn-int8"].group)

    m = pd.read_csv(ev / "masking_ci.csv")
    assert "fakecnn-int8" not in set(m.variant)                       # no masked outputs: skipped
    assert {"fakecnn", "fakevit"} <= set(m.variant)
