"""Patient-level bootstrap for T7.

Images from the same patient look alike, so confidence intervals resample
PATIENTS (with replacement), never individual images. The same resample is
applied to all 5 seeds of a model, so each interval covers the seed-averaged
result, and to raw and temperature-scaled versions alike, so their difference
gets its own interval.

Metrics here are vectorised re-implementations of the T6 definitions in
metrics.py and deferral.py (checked to give identical values in the tests).
"""
import numpy as np

from src.eval.scores import INVALID_PRED, INVALID_SCORE

N_CLASSES = 4


class PatientSampler:
    """Fast resampling of rows grouped by patient."""

    def __init__(self, patient_ids: np.ndarray, seed: int = 2026):
        codes, inverse = np.unique(patient_ids, return_inverse=True)
        self.perm = np.argsort(inverse, kind="stable")          # rows grouped by patient
        self.sizes = np.bincount(inverse, minlength=len(codes))
        self.starts = np.r_[0, np.cumsum(self.sizes)[:-1]]
        self.n_patients = len(codes)
        self.rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        chosen = self.rng.integers(0, self.n_patients, self.n_patients)
        sizes = self.sizes[chosen]
        total = sizes.sum()
        within = np.arange(total) - np.repeat(np.cumsum(sizes) - sizes, sizes)
        return self.perm[np.repeat(self.starts[chosen], sizes) + within]


def headline_metrics(y, pred, conf, u, threshold, coverage=0.8):
    """Metrics for one prediction set (all arrays already restricted to the sample).

    Invalid rows (pred == -1, from non-finite outputs) count as wrong, are always
    deferred, have no confidence (left out of ECE, never a silent failure) and
    are reported as invalid_rate.
    """
    correct = pred == y
    wrong = ~correct
    n = len(y)
    invalid = pred == INVALID_PRED
    u = np.where(invalid, INVALID_SCORE, u)

    counts = np.bincount(y, minlength=N_CLASSES)
    hits = np.bincount(y, weights=correct, minlength=N_CLASSES)
    present = counts > 0
    balanced_accuracy = float((hits[present] / counts[present]).mean())

    has_conf = np.isfinite(conf)
    c, ok = conf[has_conf], correct[has_conf]
    bins = np.clip(np.ceil(c * 15).astype(int) - 1, 0, 14)  # same (lo, hi] bins as metrics.ece
    sum_conf = np.bincount(bins, weights=c, minlength=15)
    sum_correct = np.bincount(bins, weights=ok, minlength=15)
    ece = float(np.abs(sum_correct - sum_conf).sum() / len(c)) if len(c) else np.nan

    answered = (u <= threshold) & ~invalid
    n_answered = answered.sum()
    k = max(1, int(round(coverage * n)))
    keep = np.argsort(u, kind="stable")[:k]
    return {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": balanced_accuracy,
        "ece": ece,
        "silent_failure": float((wrong & has_conf & (np.nan_to_num(conf) > 0.9)).mean()),
        "acc_at_80": float(correct[keep].mean()),
        "deployed_coverage_80": float(answered.mean()),
        "deployed_acc_80": float(correct[answered].mean()) if n_answered else np.nan,
        "errors_passed_80": float((wrong & answered).sum() / max(wrong.sum(), 1)),
        "invalid_rate": float(invalid.mean()),
    }


def coverage_curve(y, pred, u, grid):
    """Accuracy of the most-certain fraction of images, for each coverage in grid."""
    correct = (pred == y)[np.argsort(u, kind="stable")]
    cum = np.cumsum(correct)
    ks = np.maximum(1, np.round(np.asarray(grid) * len(y)).astype(int))
    return cum[ks - 1] / ks
