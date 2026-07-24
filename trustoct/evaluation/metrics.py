import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, confusion_matrix
)

@torch.no_grad()
def evaluate_classification(model, data_loader, device, classes=['CNV', 'DME', 'DRUSEN', 'NORMAL']):
    """
    Evaluates model predictions and computes a complete suite of paper metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in data_loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Core Metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)

    # Per-class Specificity calculation
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes)))
    specificities = []
    for i in range(len(classes)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    macro_spec = np.mean(specificities)

    # ROC-AUC (One-vs-Rest)
    try:
        roc_auc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except Exception:
        roc_auc_macro = 0.0

    metrics_dict = {
        'Accuracy': float(acc),
        'Precision (Macro)': float(macro_prec),
        'Recall / Sensitivity (Macro)': float(macro_rec),
        'Specificity (Macro)': float(macro_spec),
        'Macro-F1': float(macro_f1),
        'Weighted-F1': float(weighted_f1),
        'Balanced Accuracy': float(bal_acc),
        'MCC': float(mcc),
        'Cohen Kappa': float(kappa),
        'ROC-AUC (Macro)': float(roc_auc_macro)
    }

    return metrics_dict, all_labels, all_preds, all_probs, cm


def plot_confusion_matrix(cm, classes, save_path=None, title="Confusion Matrix"):
    """Plot publication-quality confusion matrix heatmap."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 12, "weight": "bold"})
    plt.title(title, fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
