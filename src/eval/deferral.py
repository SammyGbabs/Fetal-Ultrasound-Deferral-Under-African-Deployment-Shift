"""Deferral: the model answers the images it is most certain about and passes the rest to a clinician.

Thresholds are chosen on Spanish validation data to hit a target coverage
(e.g. 80% answered) and then applied unchanged to every other condition,
as a real deployment would do. On shifted data the realised coverage will
differ from the target; that difference is itself a result.
"""
import numpy as np


def threshold_for_coverage(val_uncertainty: np.ndarray, coverage: float) -> float:
    """Uncertainty value below which `coverage` of validation images fall."""
    return float(np.quantile(val_uncertainty, coverage))


def apply(y, pred, uncertainty, threshold) -> dict:
    answered = uncertainty <= threshold
    n_ans = int(answered.sum())
    return {
        "coverage": float(answered.mean()),
        "accuracy_answered": float((y[answered] == pred[answered]).mean()) if n_ans else float("nan"),
        "errors_answered_share": float(((y != pred) & answered).sum() / max((y != pred).sum(), 1)),
        "n_answered": n_ans,
    }


def accuracy_at_coverage(y, pred, uncertainty, coverage: float) -> float:
    """Oracle version: answer exactly the `coverage` most-certain images of THIS set.

    Used for the risk-coverage comparison between methods; the deployment
    version (thresholds from validation) is `apply`.
    """
    k = max(1, int(round(coverage * len(y))))
    keep = np.argsort(uncertainty, kind="stable")[:k]
    return float((y[keep] == pred[keep]).mean())
