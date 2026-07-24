import numpy as np
import matplotlib.pyplot as plt

def compute_calibration_metrics(y_true, y_probs, num_bins=10):
    """
    Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    and Brier Score for multi-class classification.
    """
    y_preds = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)
    accuracies = (y_preds == y_true)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    mce = 0.0

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples falling into current confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])

            gap = np.abs(accuracy_in_bin - avg_confidence_in_bin)
            ece += gap * prop_in_bin
            mce = max(mce, gap)

            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
            bin_counts.append(np.sum(in_bin))
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)
            bin_counts.append(0)

    # Compute Multi-class Brier Score
    num_classes = y_probs.shape[1]
    one_hot_true = np.eye(num_classes)[y_true]
    brier_score = np.mean(np.sum((y_probs - one_hot_true) ** 2, axis=1))

    results = {
        'ECE': float(ece),
        'MCE': float(mce),
        'Brier_Score': float(brier_score),
        'bin_accs': bin_accs,
        'bin_confs': bin_confs,
        'bin_counts': bin_counts,
        'bin_boundaries': bin_boundaries
    }
    return results


def plot_reliability_diagram(results, model_name="TrustOCT", save_path=None):
    """
    Plots a publication-grade Reliability Diagram (Confidence vs Accuracy).
    """
    bin_accs = results['bin_accs']
    bin_confs = results['bin_confs']
    num_bins = len(bin_accs)
    bin_centers = np.linspace(0.05, 0.95, num_bins)

    plt.figure(figsize=(7, 6))
    
    # Plot Perfect Calibration Baseline
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    
    # Plot Confidence-Accuracy Gap Bars
    plt.bar(bin_centers, bin_accs, width=0.08, alpha=0.7, color='#2b5c8f', edgecolor='black', label='Model Accuracy')
    
    # Highlight Calibration Gap
    plt.bar(bin_centers, np.abs(np.array(bin_accs) - bin_centers), bottom=np.minimum(bin_accs, bin_centers),
            width=0.08, alpha=0.3, color='#e74c3c', edgecolor='red', label='Calibration Gap')

    plt.xlabel('Confidence', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title(f'Reliability Diagram ({model_name})\nECE: {results["ECE"]*100:.2f}% | Brier Score: {results["Brier_Score"]:.4f}',
              fontsize=13, fontweight='bold')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
