import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def plot_fairness(res_base, res_adv, results_dir):
    metric_names = [("model_dp", "Demographic Parity"), ("model_eo", "Equal Opportunity"), ("model_eod", "Equalized Odds")]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    plt.figure(figsize=(10, 12))
    for i, (col, name) in enumerate(metric_names):
        plt.subplot(3, 1, i+1)
        plt.plot(res_base["dataset_dp"], res_base[col], 'o-', color=colors[i], alpha=0.7, label=f'Original: {name}')
        plt.plot(res_adv["dataset_dp"], res_adv[col], 's-', color=colors[i], linestyle="--", alpha=0.7, label=f'Adversarial: {name}')
        corr_base, p_base = pearsonr(res_base["dataset_dp"], res_base[col])
        corr_adv, p_adv = pearsonr(res_adv["dataset_dp"], res_adv[col])
        plt.annotate(f"Original r={corr_base:.2f}, p={p_base:.2g}", (0.02, 0.95), xycoords='axes fraction', color=colors[i])
        plt.annotate(f"Adversarial r={corr_adv:.2f}, p={p_adv:.2g}", (0.02, 0.88), xycoords='axes fraction', color=colors[i], fontweight='bold')
        plt.xlabel("Dataset Demographic Parity (Unfairness)")
        plt.ylabel(f"Model {name} (Difference)")
        plt.title(f"Fairness Metric: {name} (Original vs. Adversarial)")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "adversarial_fairness_comparison.png"))
    plt.close()

def plot_efficiency(res_base, eff_base_df, res_adv, eff_adv_df, results_dir):
    eff_metric_names = [
        ("recall", "Recall"), ("precision", "Precision"), ("f1", "F1-score"),
        ("roc_auc", "ROC–AUC"), ("informedness", "Informedness"),
        ("markedness", "Markedness"), ("mcc", "MCC")
    ]
    eff_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:olive']
    plt.figure(figsize=(14, 18))
    for i, (col, name) in enumerate(eff_metric_names):
        plt.subplot(4, 2, i+1)
        plt.plot(res_base["dataset_dp"], eff_base_df[col], 'o-', color=eff_colors[i], label=f"Original: {name}")
        plt.plot(res_adv["dataset_dp"], eff_adv_df[col], 's-', color=eff_colors[i], linestyle="--", label=f"Adversarial: {name}")
        corr_base, p_base = pearsonr(res_base["dataset_dp"], eff_base_df[col])
        corr_adv, p_adv = pearsonr(res_adv["dataset_dp"], eff_adv_df[col])
        plt.annotate(f"Original r={corr_base:.2f}, p={p_base:.2g}", (0.02, 0.95), xycoords='axes fraction', color=eff_colors[i])
        plt.annotate(f"Adversarial r={corr_adv:.2f}, p={p_adv:.2g}", (0.02, 0.88), xycoords='axes fraction', color=eff_colors[i], fontweight='bold')
        plt.xlabel("Dataset Demographic Parity (Unfairness)")
        plt.ylabel(name)
        plt.title(f"Efficiency Metric: {name} (Original vs. Adversarial)")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "adversarial_efficiency_comparison.png"))
    plt.close()

def plot_weights(weights_baseline, weights_adv, corr, corr_p, t_p, results_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(weights_baseline, marker='o', label='Original')
    plt.plot(weights_adv, marker='s', label='Adversarial')
    plt.title(
        f"First-layer weights at bias=0\n"
        f"Pearson r={corr:.3f}, p={corr_p:.2g}; paired t-test p={t_p:.2g}"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "main_model_weights_bias0.png"))
    plt.close()