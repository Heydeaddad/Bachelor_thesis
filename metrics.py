import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score, matthews_corrcoef

def fairness_metrics(y_true, y_pred, protected):
    dp = []
    for group in [0, 1]:
        mask = protected == group
        dp.append(np.mean(y_pred[mask]))
    dp_diff = abs(dp[1] - dp[0])
    tpr = []
    for group in [0, 1]:
        mask = protected == group
        cm = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1])
        tp = cm[1, 1] if cm.shape[0] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 else 0
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else np.nan)
    eo_diff = abs(tpr[1] - tpr[0])
    fpr = []
    for group in [0, 1]:
        mask = protected == group
        cm = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1])
        fp = cm[0, 1] if cm.shape[0] > 0 else 0
        tn = cm[0, 0] if cm.shape[0] > 0 else 0
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else np.nan)
    eod_diff = (abs(tpr[1] - tpr[0]) + abs(fpr[1] - fpr[0])) / 2
    return dp_diff, eo_diff, eod_diff

def dataset_demographic_parity(y, protected):
    dp = []
    for group in [0, 1]:
        mask = protected == group
        dp.append(np.mean(y[mask]))
    return abs(dp[1] - dp[0])

def informedness(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2,2): return np.nan
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensitivity + specificity - 1

def markedness(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2,2): return np.nan
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    return precision + npv - 1

def efficiency_metrics(y_true, y_pred, y_pred_proba):
    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "informedness": informedness(y_true, y_pred),
        "markedness": markedness(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred)
    }