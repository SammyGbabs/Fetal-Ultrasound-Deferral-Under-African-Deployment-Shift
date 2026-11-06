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
"""
import numpy as np
from scipy.special import log_softmax, logsumexp
from sklearn.covariance import LedoitWolf

EPS = 1e-12


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
    lp = log_softmax(logits / T, axis=1)
    return lp.argmax(1), log_one_minus_max(lp)


def predictive_entropy(logits, T=1.0):
    """Uncertainty = log(entropy) of the predicted probabilities."""
    lp = log_softmax(logits / T, axis=1)
    return lp.argmax(1), log_entropy(lp)


def energy(logits, T=1.0):
    """Energy score (Liu et al., 2020): -T * logsumexp(logits / T). Higher = less familiar."""
    return logits.argmax(1), -T * logsumexp(logits / T, axis=1)


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
    lp = log_softmax(mc_logits / T, axis=2)               # (N, passes, C)
    log_mean = _mean_log_probs(lp, axis=1)
    total = np.exp(log_entropy(log_mean))
    mutual_info = total - np.exp(log_entropy(lp)).mean(1)
    return log_mean.argmax(1), log_entropy(log_mean), mutual_info


def ensemble(logits_per_seed: list, temps: list | None = None):
    """Deep ensemble from the 5 seed runs of one variant (same condition, same row order)."""
    temps = temps or [1.0] * len(logits_per_seed)
    lp = np.stack([log_softmax(l / t, axis=1) for l, t in zip(logits_per_seed, temps)])  # (S, N, C)
    log_mean = _mean_log_probs(lp, axis=0)
    total = np.exp(log_entropy(log_mean))
    mutual_info = total - np.exp(log_entropy(lp)).mean(0)
    return log_mean.argmax(1), log_entropy(log_mean), mutual_info


class Mahalanobis:
    """Class-conditional Gaussians with a shared covariance, fitted on Spanish training features.

    Uncertainty = distance to the nearest class mean. Ledoit-Wolf shrinkage keeps the
    covariance stable when the feature dimension is large relative to 2,992 images.
    """

    def fit(self, features: np.ndarray, y: np.ndarray, n_classes: int = 4):
        self.means = np.stack([features[y == c].mean(0) for c in range(n_classes)])
        centred = features - self.means[y]
        self.precision = LedoitWolf().fit(centred).precision_
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        d = []
        for m in self.means:
            diff = features - m
            d.append(np.einsum("ij,jk,ik->i", diff, self.precision, diff))
        return np.min(np.stack(d, 1), axis=1)
