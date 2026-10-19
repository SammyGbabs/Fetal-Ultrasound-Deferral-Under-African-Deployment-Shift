"""Uncertainty scores. Convention: HIGHER score = MORE uncertain = defer first.

Each method returns (prediction, uncertainty) for every image, because some
methods (MC dropout, ensemble) also change the prediction itself.
"""
import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.covariance import LedoitWolf

EPS = 1e-12


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(p + EPS)).sum(axis=-1)


def msp(logits, T=1.0):
    """Maximum softmax probability: uncertainty = 1 - top probability."""
    p = softmax(logits / T, axis=1)
    return p.argmax(1), 1.0 - p.max(1)


def predictive_entropy(logits, T=1.0):
    p = softmax(logits / T, axis=1)
    return p.argmax(1), entropy(p)


def energy(logits, T=1.0):
    """Energy score (Liu et al., 2020): -T * logsumexp(logits / T). Higher = less familiar."""
    return logits.argmax(1), -T * logsumexp(logits / T, axis=1)


def mc_dropout(mc_logits, T=1.0):
    """20 stochastic passes. Prediction from the mean probability.

    Returns the prediction, total entropy of the mean, and mutual information
    (the part of the uncertainty due to disagreement between passes).
    Note: for frozen backbones the only dropout is in the head, so this
    measures head uncertainty only.
    """
    p = softmax(mc_logits / T, axis=2)          # (N, passes, C)
    mean = p.mean(1)
    total = entropy(mean)
    mutual_info = total - entropy(p).mean(1)
    return mean.argmax(1), total, mutual_info


def ensemble(logits_per_seed: list, temps: list | None = None):
    """Deep ensemble from the 5 seed runs of one variant (same condition, same row order)."""
    temps = temps or [1.0] * len(logits_per_seed)
    p = np.stack([softmax(l / t, axis=1) for l, t in zip(logits_per_seed, temps)])  # (S, N, C)
    mean = p.mean(0)
    total = entropy(mean)
    mutual_info = total - entropy(p).mean(0)
    return mean.argmax(1), total, mutual_info


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
