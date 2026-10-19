"""Metrics. All functions take plain arrays, so they work per run, per seed or per bootstrap sample."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def accuracy(y, pred):
    return float((y == pred).mean())


def balanced_accuracy(y, pred):
    """Mean recall over the classes present in y (so Ghana/Uganda are scored on 3 classes)."""
    return float(balanced_accuracy_score(y, pred))


def macro_f1(y, pred):
    return float(f1_score(y, pred, labels=np.unique(y), average="macro", zero_division=0))


def ece(probs, y, n_bins=15):
    """Expected calibration error on the top-label confidence, 15 equal-width bins."""
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def brier(probs, y):
    onehot = np.eye(probs.shape[1])[y]
    return float(((probs - onehot) ** 2).sum(1).mean())


def silent_failure_rate(probs, y, threshold=0.9):
    """Share of all predictions that are wrong yet made with confidence above the threshold."""
    conf = probs.max(1)
    wrong = probs.argmax(1) != y
    return float((wrong & (conf > threshold)).mean())


def risk_coverage(y, pred, uncertainty):
    """Accuracy of the answered cases as coverage goes from 1 image to all images.

    Images are answered from most to least certain. Returns (coverage, accuracy) arrays.
    """
    order = np.argsort(uncertainty, kind="stable")
    correct = (y[order] == pred[order]).astype(float)
    n = len(y)
    coverage = np.arange(1, n + 1) / n
    acc = np.cumsum(correct) / np.arange(1, n + 1)
    return coverage, acc


def aurc(y, pred, uncertainty):
    """Area under the risk-coverage curve (risk = error rate). Lower is better."""
    _, acc = risk_coverage(y, pred, uncertainty)
    return float((1 - acc).mean())


def ood_auroc(u_in, u_out):
    """How well an uncertainty score separates shifted (out) from familiar (in) images. 0.5 = chance."""
    labels = np.r_[np.zeros(len(u_in)), np.ones(len(u_out))]
    return float(roc_auc_score(labels, np.r_[u_in, u_out]))
