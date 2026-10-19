"""Temperature scaling, fitted on Spanish validation data only.

One number T per run rescales the logits (logits / T). T > 1 softens
confidence, T < 1 sharpens it. Because validation accuracy is close to 100%,
the fitted T may sharpen confidence; every metric is therefore reported both
with and without it.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def nll(logits: np.ndarray, y: np.ndarray, T: float = 1.0) -> float:
    return float(-log_softmax(logits / T, axis=1)[np.arange(len(y)), y].mean())


def fit_temperature(val_logits: np.ndarray, val_y: np.ndarray,
                    bounds: tuple = (0.05, 20.0)) -> float:
    """Minimise validation NLL over log T (bounded, so T cannot run off to 0)."""
    res = minimize_scalar(lambda lt: nll(val_logits, val_y, np.exp(lt)),
                          bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
    return float(np.exp(res.x))


def probs(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Softmax over the last axis, after temperature."""
    return softmax(logits / T, axis=-1)
