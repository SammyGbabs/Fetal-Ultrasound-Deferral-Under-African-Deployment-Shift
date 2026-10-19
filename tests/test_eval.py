"""Checks for the T6 evaluation code. Run:  python -m pytest tests/ -q"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval import calibration, deferral, metrics, scores  # noqa: E402

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


def test_end_to_end_on_fake_outputs(tmp_path):
    fake = tmp_path / "outputs"
    subprocess.run([sys.executable, ROOT / "scripts/make_fake_outputs.py", fake], check=True)
    subprocess.run([sys.executable, ROOT / "scripts/validate_outputs.py", fake], check=True)
    eval_dir = tmp_path / "eval"
    subprocess.run([sys.executable, ROOT / "scripts/run_eval.py",
                    "--outputs", fake, "--eval-dir", eval_dir], check=True)
    df = pd.read_csv(eval_dir / "per_run_metrics.csv")
    # 2 variants x 5 seeds x 16 test conditions x 8 scores x 2 calibrations
    assert len(df) == 2 * 5 * 16 * 8 * 2
    clean = df[(df.score == "msp") & (df.calib == "raw")]
    spain = clean[clean.condition == "spain_test"].accuracy.mean()
    africa = clean[clean.condition == "africa_test"].accuracy.mean()
    africa_q10 = clean[clean.condition == "africa_test_q10"].accuracy.mean()
    assert spain > africa > africa_q10                        # the fake shifts are detected
    maha = df[(df.score == "mahalanobis") & (df.condition == "africa_test")].ood_auroc.mean()
    assert maha > 0.9                                         # fake African features are shifted
