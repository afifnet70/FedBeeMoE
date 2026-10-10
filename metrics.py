import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import CFG

METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Weighted_F1",
    "AUC",
    "MCC",
    "CSI",
    "Kappa",
    "Specificity",
]
NB = CFG["n_bins"]


def compute_metrics(y, prob, thr=0.5):
    y = np.asarray(y)
    prob = np.asarray(prob)
    pred = (prob >= np.asarray(thr) - 1e-9).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y, prob)
        pr = average_precision_score(y, prob)
    except Exception:
        auc = pr = float("nan")
    return dict(
        Accuracy=accuracy_score(y, pred),
        Precision=precision_score(y, pred, zero_division=0),
        Recall=recall_score(y, pred, zero_division=0),
        F1=f1_score(y, pred, zero_division=0),
        Weighted_F1=f1_score(y, pred, average="weighted", zero_division=0),
        AUC=auc,
        MCC=matthews_corrcoef(y, pred) if len(set(pred)) > 1 else 0.0,
        CSI=tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0,
        Kappa=cohen_kappa_score(y, pred) if len(set(pred)) > 1 else 0.0,
        Specificity=tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        PR_AUC=pr,
        TP=int(tp),
        FP=int(fp),
        FN=int(fn),
        TN=int(tn),
    )


def client_hist(y, p):
    b = np.minimum((np.asarray(p) * NB).astype(int), NB - 1)
    return np.bincount(b[np.asarray(y) == 1], minlength=NB), np.bincount(b[np.asarray(y) == 0], minlength=NB)


def select_from_hists(hp, hn):
    tp = hp[::-1].cumsum()[::-1]
    fp = hn[::-1].cumsum()[::-1]
    P, N = hp.sum(), hn.sum()
    fn, tn = P - tp, N - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).astype(float)
    mcc = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0.0)
    lo, hi = int(0.1 * NB), int(0.9 * NB)
    k = lo + int(np.argmax(mcc[lo : hi + 1]))
    tpr = np.concatenate([tp / max(P, 1), [0]])
    fpr = np.concatenate([fp / max(N, 1), [0]])
    auc = float(np.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + tpr[1:]) / 2)) if P > 0 and N > 0 else 0.5
    return dict(score=float(mcc[k] + auc), thr=k / NB, mcc=float(mcc[k]), auc=auc)
