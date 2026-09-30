"""Evaluation script for RQ1, RQ2, RQ3, and RQ4.
Computes Pearson r, RMSE, Macro-F1, Accuracy, Weighted-F1,
Same-Valence Confusion (SVC), paired bootstrap CIs, McNemar tests,
Spearman rho intensity correlation, and formats LaTeX tables.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2, pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score

from config import (
    RESULTS_DIR,
    SVC_GROUPS_EMOWOZ,
    SVC_GROUPS_ENVENT,
    THETA_GRID,
    TO_EMOWOZ,
    TO_ENVENT,
)


def compute_svc(y_true, y_pred, svc_groups):
    """Compute Same-Valence Confusion (SVC) according to Eq. (5) of the paper."""
    svc_scores = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for group_name, labels in svc_groups.items():
        # Mask instances whose ground-truth label belongs to the valence group
        mask = np.isin(y_true, labels)
        if np.sum(mask) == 0:
            continue
        group_true = y_true[mask]
        group_pred = y_pred[mask]
        # Count how many are predicted as another emotion in the same group
        confused = np.isin(group_pred, labels) & (group_pred != group_true)
        svc = np.mean(confused)
        svc_scores.append(svc)

    return float(np.mean(svc_scores)) if svc_scores else 0.0


def paired_bootstrap_ci(y_true, y_pred, metric_fn, n_bootstraps=1000, alpha=0.05, seed=42):
    """Compute 95% confidence interval via paired bootstrap."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        scores.append(metric_fn(np.array(y_true)[idx], np.array(y_pred)[idx]))
    lower = np.percentile(scores, 100 * (alpha / 2))
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return lower, upper


def run_mcnemar_test(y_true, pred_ours, pred_base):
    """Run McNemar's test with continuity correction."""
    correct_ours = np.array(y_true) == np.array(pred_ours)
    correct_base = np.array(y_true) == np.array(pred_base)

    b = float(np.sum(correct_ours & (~correct_base)))
    c = float(np.sum((~correct_ours) & correct_base))

    if b + c == 0:
        return 1.0
    # McNemar statistic with Edward's continuity correction
    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    p_value = float(chi2.sf(stat, df=1))
    return p_value


def evaluate_rq1(envent_preds, emowoz_preds):
    """RQ1: Appraisal estimation intrinsic evaluation."""
    # enVENT test
    d_pred = envent_preds["d"]
    d_true = envent_preds["target_d"]
    r_d, _ = pearsonr(d_pred, d_true)
    rmse_d = np.sqrt(np.mean((d_pred - d_true) ** 2))

    mask = envent_preds["target_mask"] == 1.0
    p_pred = envent_preds["p"][mask]
    p_true = envent_preds["target_p"][mask]
    r_p, _ = pearsonr(p_pred, p_true)
    rmse_p = np.sqrt(np.mean((p_pred - p_true) ** 2))

    agent_pred = np.argmax(envent_preds["pi"], axis=-1)
    agent_true = envent_preds["target_agent"]
    f1_agent_envent = f1_score(agent_true, agent_pred, average="macro")

    # EmoWOZ non-neutral turns
    non_neutral = emowoz_preds["valence"] != 0
    emowoz_d = emowoz_preds["d"][non_neutral]
    emowoz_val_true = emowoz_preds["valence"][non_neutral]
    val_pred = np.where(emowoz_d >= 0, 1, -1)
    acc_val_emowoz = accuracy_score(emowoz_val_true, val_pred)

    emowoz_agent_pred = np.argmax(emowoz_preds["pi"][non_neutral], axis=-1)
    emowoz_agent_true = emowoz_preds["agent_idx"][non_neutral]
    f1_agent_emowoz = f1_score(emowoz_agent_true, emowoz_agent_pred, average="macro")

    results = {
        "envent_d_r": float(r_d),
        "envent_d_rmse": float(rmse_d),
        "envent_p_r": float(r_p),
        "envent_p_rmse": float(rmse_p),
        "envent_agent_f1": float(f1_agent_envent),
        "emowoz_valence_acc": float(acc_val_emowoz),
        "emowoz_agent_f1": float(f1_agent_emowoz),
    }
    print("[RQ1 Results]:")
    for k, v in results.items():
        print(f"  {k}: {v:.3f}")
    return results


def evaluate_recognition(y_true, y_pred, dataset="envent"):
    """Evaluate classification metrics (Macro-F1, Acc, SVC, Weighted-F1)."""
    labels = sorted(list(set(y_true)))
    acc = accuracy_score(y_true, y_pred)

    if dataset == "envent":
        mf1 = f1_score(y_true, y_pred, average="macro")
        wf1 = f1_score(y_true, y_pred, average="weighted")
        svc = compute_svc(y_true, y_pred, SVC_GROUPS_ENVENT)
    else:
        # EmoWOZ: macro-F1 over the 5 emotion classes without neutral (line 310 of paper)
        emo_classes = [c for c in labels if c != "neutral"]
        mf1 = f1_score(y_true, y_pred, labels=emo_classes, average="macro")
        wf1 = f1_score(y_true, y_pred, average="weighted")
        svc = compute_svc(y_true, y_pred, SVC_GROUPS_EMOWOZ)

    return {
        "macro_f1": float(mf1),
        "accuracy": float(acc),
        "weighted_f1": float(wf1),
        "svc": float(svc),
    }


def find_best_theta(val_d, val_p, val_pi, val_labels, engine):
    """Grid search for optimal theta on validation split."""
    best_theta = 0.3
    best_f1 = -1.0
    for theta in THETA_GRID:
        preds = []
        for i in range(len(val_d)):
            lbl, _, _, _, _, _, _ = engine.decide(val_d[i], val_p[i], val_pi[i], theta=theta, dataset="envent")
            preds.append(lbl)
        f1 = f1_score(val_labels, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_theta = theta
    return best_theta, best_f1
