from typing import Dict
import numpy as np
from sklearn.metrics import (
    recall_score,
    roc_auc_score,
    f1_score,
    accuracy_score,
)


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    uar = float(recall_score(labels, preds, average="macro", zero_division=0))
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")
    f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    acc = float(accuracy_score(labels, preds))
    return {"uar": uar, "auc_roc": auc, "f1": f1, "accuracy": acc}
