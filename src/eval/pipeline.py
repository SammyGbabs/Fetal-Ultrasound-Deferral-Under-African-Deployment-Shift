"""End-to-end evaluation of saved outputs.

Two steps, kept separate so T7 can reuse them:
1. `per_image(...)`: for one run and one condition, every image's prediction,
   probabilities and uncertainty under every method, with and without
   temperature scaling.
2. `summarize(...)`: metrics from those per-image results on any subset of
   rows (all images, one country, one machine, or a bootstrap resample).

Predictors (what makes the decision):  single, mc, ensemble
Calibration:                           raw, ts (temperature-scaled)
Uncertainty scores (what defers):      msp, entropy, energy, mc_entropy,
                                       mc_mi, mahalanobis, ens_entropy, ens_mi
"""
from dataclasses import dataclass, field

import numpy as np

from src.eval import calibration, deferral, metrics, scores

COVERAGES = (0.8, 0.9)

# which predictor each uncertainty score belongs to
SCORE_PREDICTOR = {
    "msp": "single", "entropy": "single", "energy": "single", "mahalanobis": "single",
    "mc_entropy": "mc", "mc_mi": "mc",
    "ens_entropy": "ensemble", "ens_mi": "ensemble",
}


@dataclass
class RunFit:
    """Everything fitted on Spanish data for one run: temperature and Mahalanobis model."""
    T: float
    maha: scores.Mahalanobis
    val_u: dict = field(default_factory=dict)   # (score, calib) -> validation uncertainties


def fit_run(val: dict, train: dict) -> RunFit:
    T = calibration.fit_temperature(val["logits"], val["y_true"])
    maha = scores.Mahalanobis().fit(train["features"], train["y_true"])
    return RunFit(T=T, maha=maha)


def per_image(out: dict, fit: RunFit, ens_logits: list | None = None,
              ens_temps: list | None = None) -> dict:
    """All per-image quantities for one run on one condition.

    ens_logits: logits of the 5 seed runs of this variant on the same condition
                (needed only for the ensemble methods).
    """
    res = {"y": out["y_true"].astype(int)}
    for calib, T in (("raw", 1.0), ("ts", fit.T)):
        p = calibration.probs(out["logits"], T)
        res[("single", calib)] = (p.argmax(1), p)
        res[("msp", calib)] = scores.msp(out["logits"], T)[1]
        res[("entropy", calib)] = scores.predictive_entropy(out["logits"], T)[1]
        res[("energy", calib)] = scores.energy(out["logits"], T)[1]

        mc_p = calibration.probs(out["mc_logits"], T).mean(1)
        pred, total, mi = scores.mc_dropout(out["mc_logits"], T)
        res[("mc", calib)] = (pred, mc_p)
        res[("mc_entropy", calib)] = total
        res[("mc_mi", calib)] = mi

        if ens_logits is not None:
            temps = ens_temps if calib == "ts" else None
            pred, total, mi = scores.ensemble(ens_logits, temps)
            ens_p = np.mean([calibration.probs(l, t) for l, t in
                             zip(ens_logits, temps or [1.0] * len(ens_logits))], axis=0)
            res[("ensemble", calib)] = (pred, ens_p)
            res[("ens_entropy", calib)] = total
            res[("ens_mi", calib)] = mi

        res[("mahalanobis", calib)] = fit.maha.score(out["features"])  # independent of T
    return res


def summarize(res: dict, idx: np.ndarray | None = None, val_res: dict | None = None,
              ref_res: dict | None = None) -> list[dict]:
    """Metrics for every (score, calib) on rows `idx` (default: all rows).

    val_res: per_image results on spain_val, for deployment thresholds.
    ref_res: per_image results on clean spain_test, for OOD AUROC (familiar images).
    """
    y_all = res["y"]
    idx = np.arange(len(y_all)) if idx is None else idx
    y = y_all[idx]
    rows = []
    for score, predictor in SCORE_PREDICTOR.items():
        for calib in ("raw", "ts"):
            if (score, calib) not in res:
                continue                     # e.g. ensemble when seeds are missing
            pred_all, p_all = res[(predictor, calib)]
            pred, p, u = pred_all[idx], p_all[idx], res[(score, calib)][idx]
            row = {
                "score": score, "predictor": predictor, "calib": calib, "n": len(y),
                "accuracy": metrics.accuracy(y, pred),
                "balanced_accuracy": metrics.balanced_accuracy(y, pred),
                "macro_f1": metrics.macro_f1(y, pred),
                "ece": metrics.ece(p, y),
                "brier": metrics.brier(p, y),
                "silent_failure": metrics.silent_failure_rate(p, y),
                "aurc": metrics.aurc(y, pred, u),
            }
            for c in COVERAGES:
                tag = int(c * 100)
                row[f"acc_at_{tag}"] = deferral.accuracy_at_coverage(y, pred, u, c)
                if val_res is not None:
                    thr = deferral.threshold_for_coverage(val_res[(score, calib)], c)
                    d = deferral.apply(y, pred, u, thr)
                    row[f"deployed_coverage_{tag}"] = d["coverage"]
                    row[f"deployed_acc_{tag}"] = d["accuracy_answered"]
                    row[f"errors_passed_{tag}"] = d["errors_answered_share"]
            if ref_res is not None:
                row["ood_auroc"] = metrics.ood_auroc(ref_res[(score, calib)], u)
            rows.append(row)
    return rows
