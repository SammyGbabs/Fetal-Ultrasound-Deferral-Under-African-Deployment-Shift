"""Uncertainty scores. Convention: HIGHER score = MORE uncertain = defer first.

Each method returns (prediction, uncertainty) for every image, because some
methods (MC dropout, ensemble) also change the prediction itself.

Numerical stability: with a very low temperature (e.g. T = 0.05), almost
every softmax probability rounds to exactly 1 or 0, so "1 - max probability"
and entropy collapse to exactly 0 for many images at once. Their order is
then decided by rounding, not by the model. All scores are therefore computed
from log-probabilities and reported on a log scale:
    msp:      log(1 - max probability)
    entropy:  log(entropy)
Both are increasing functions of the original scores, so every image keeps
exactly the same rank, deferral thresholds pick the same images and all
rank-based metrics (risk-coverage, AURC, OOD AUROC) are unchanged; only the
ties caused by rounding disappear.

Non-finite outputs: a row with any NaN or inf in its logits (for MC dropout:
in any pass; for the ensemble: in any seed) gives no usable prediction. Such
a row gets prediction INVALID_PRED (-1, so it always counts as wrong) and
uncertainty INVALID_SCORE (the largest finite float, so every deferral
threshold sends it to a clinician; a finite value keeps sklearn's AUROC
working). This happens with some FP16 runs, whose activations overflow.
"""
import numpy as np
from scipy.special import log_softmax, logsumexp
from sklearn.covariance import LedoitWolf

EPS = 1e-12
INVALID_PRED = -1
INVALID_SCORE = float(np.finfo(np.float64).max)


def invalid_rows(a: np.ndarray) -> np.ndarray:
    """True for every row (first axis) that has any non-finite value."""
    a = np.asarray(a)
    return ~np.isfinite(a.reshape(len(a), -1)).all(axis=1)


def _clean(a: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Copy with invalid rows set to 0, so the maths below stays finite."""
    if not bad.any():
        return a
    a = np.array(a, dtype=np.float64, copy=True)
    a[bad] = 0.0
    return a


def _mark(bad: np.ndarray, pred: np.ndarray, *uncertainties):
    """Invalid rows: prediction -1 (always wrong), uncertainty INVALID_SCORE (always deferred)."""
    if not bad.any():
        return (pred, *uncertainties)
    pred = np.where(bad, INVALID_PRED, pred)
    return (pred, *(np.where(bad, INVALID_SCORE, u) for u in uncertainties))


def entropy(p: np.ndarray) -> np.ndarray:
    """Plain entropy of probabilities (used where the value itself is needed)."""
    return -(p * np.log(p + EPS)).sum(axis=-1)


def log_one_minus_max(log_p: np.ndarray) -> np.ndarray:
    """log(1 - max probability), computed without forming 1 - p (no rounding to 0)."""
    top = log_p.argmax(-1)
    others = np.where(np.eye(log_p.shape[-1], dtype=bool)[top], -np.inf, log_p)
    return logsumexp(others, axis=-1)


def log_entropy(log_p: np.ndarray) -> np.ndarray:
    """log(entropy) from log-probabilities: log sum_j p_j * (-log p_j)."""
    valid = np.isfinite(log_p) & (log_p < 0)
    safe = np.where(valid, log_p, -1.0)
    terms = np.where(valid, safe + np.log(-safe), -np.inf)
    return logsumexp(terms, axis=-1)


def msp(logits, T=1.0):
    """Maximum softmax probability. Uncertainty = log(1 - top probability)."""
    bad = invalid_rows(logits)
    lp = log_softmax(_clean(logits, bad) / T, axis=1)
    return _mark(bad, lp.argmax(1), log_one_minus_max(lp))


def predictive_entropy(logits, T=1.0):
    """Uncertainty = log(entropy) of the predicted probabilities."""
    bad = invalid_rows(logits)
    lp = log_softmax(_clean(logits, bad) / T, axis=1)
    return _mark(bad, lp.argmax(1), log_entropy(lp))


def energy(logits, T=1.0):
    """Energy score (Liu et al., 2020): -T * logsumexp(logits / T). Higher = less familiar."""
    bad = invalid_rows(logits)
    logits = _clean(logits, bad)
    return _mark(bad, logits.argmax(1), -T * logsumexp(logits / T, axis=1))


def _mean_log_probs(log_p: np.ndarray, axis: int) -> np.ndarray:
    """log of the mean probability over `axis`, computed in log space."""
    return logsumexp(log_p, axis=axis) - np.log(log_p.shape[axis])


def mc_dropout(mc_logits, T=1.0):
    """20 stochastic passes. Prediction from the mean probability.

    Returns the prediction, log(total entropy of the mean), and mutual
    information (the part of the uncertainty due to disagreement between
    passes). Note: for frozen backbones the only dropout is in the head, so
    this measures head uncertainty only.
    """
    bad = invalid_rows(mc_logits)                         # any pass non-finite
    lp = log_softmax(_clean(mc_logits, bad) / T, axis=2)  # (N, passes, C)
    log_mean = _mean_log_probs(lp, axis=1)
    total = np.exp(log_entropy(log_mean))
    mutual_info = total - np.exp(log_entropy(lp)).mean(1)
    return _mark(bad, log_mean.argmax(1), log_entropy(log_mean), mutual_info)


def ensemble(logits_per_seed: list, temps: list | None = None):
    """Deep ensemble from the 5 seed runs of one variant (same condition, same row order)."""
    temps = temps or [1.0] * len(logits_per_seed)
    bad = np.any([invalid_rows(l) for l in logits_per_seed], axis=0)   # any seed non-finite
    lp = np.stack([log_softmax(_clean(l, bad) / t, axis=1)
                   for l, t in zip(logits_per_seed, temps)])            # (S, N, C)
    log_mean = _mean_log_probs(lp, axis=0)
    total = np.exp(log_entropy(log_mean))
    mutual_info = total - np.exp(log_entropy(lp)).mean(0)
    return _mark(bad, log_mean.argmax(1), log_entropy(log_mean), mutual_info)


class Mahalanobis:
    """Class-conditional Gaussians with a shared covariance, fitted on Spanish training features.

    Uncertainty = distance to the nearest class mean. Ledoit-Wolf shrinkage keeps the
    covariance stable when the feature dimension is large relative to 2,992 images.
    Rows with non-finite features are left out of the fit and scored INVALID_SCORE.
    """

    def fit(self, features: np.ndarray, y: np.ndarray, n_classes: int = 4):
        ok = ~invalid_rows(features)
        features, y = features[ok], y[ok]
        self.means = np.stack([features[y == c].mean(0) for c in range(n_classes)])
        centred = features - self.means[y]
        self.precision = LedoitWolf().fit(centred).precision_
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        bad = invalid_rows(features)
        features = _clean(features, bad)
        d = []
        for m in self.means:
            diff = features - m
            d.append(np.einsum("ij,jk,ik->i", diff, self.precision, diff))
        return np.where(bad, INVALID_SCORE, np.min(np.stack(d, 1), axis=1))
