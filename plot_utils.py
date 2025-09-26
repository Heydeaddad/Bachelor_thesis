import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kendalltau
from sklearn.linear_model import TheilSenRegressor
from sklearn.utils import resample



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
    plt.figure(figsize=(8, 6))
    plt.scatter(weights_baseline, weights_adv, alpha=0.7, color='purple')
    plt.xlabel("Baseline Model Weights")
    plt.ylabel("Adversarial Model Weights")
    plt.title("Comparison of First Layer Weights (Baseline vs. Adversarial)")
    plt.annotate(f"Pearson r={corr:.2f}, p={corr_p:.3g}", (0.05, 0.92), xycoords='axes fraction', fontsize=12)
    plt.annotate(f"Paired t-test p={t_p:.3g}", (0.05, 0.86), xycoords='axes fraction', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "main_model_weights_bias0.png"))
    plt.close()



def plot_fairness_correlation_kendall_theilsen(res_base, results_dir):
    """
    Plots correlation between dataset fairness and base model fairness for each fairness metric,
    with Kendall's tau and Theil–Sen regression line (with shaded 95% CI for the slope).
    """
    metric_names = [
        ("model_dp", "Demographic Parity"),
        ("model_eo", "Equal Opportunity"),
        ("model_eod", "Equalized Odds"),
    ]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    plt.figure(figsize=(10, 14))
    for i, (col, name) in enumerate(metric_names):
        plt.subplot(3, 1, i+1)
        x = np.array(res_base["dataset_dp"])
        y = np.array(res_base[col])

        # Scatter plot
        plt.scatter(x, y, color=colors[i], alpha=0.7, label=f'{name} points')

        # Theil–Sen regression
        X = x.reshape(-1, 1)
        model = TheilSenRegressor(random_state=42)
        model.fit(X, y)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = model.predict(x_fit.reshape(-1, 1))
        plt.plot(x_fit, y_fit, color='red', label='Theil–Sen regression')

        # Bootstrap 95% CI for slope
        slopes = []
        n_bootstrap = 1000
        rng = np.random.default_rng(42)
        for _ in range(n_bootstrap):
            idx = rng.choice(len(x), size=len(x), replace=True)
            x_bs, y_bs = x[idx], y[idx]
            model_bs = TheilSenRegressor(random_state=42)
            try:
                model_bs.fit(x_bs.reshape(-1, 1), y_bs)
                slopes.append(model_bs.coef_[0])
            except Exception:
                continue
        if slopes:
            lower, upper = np.percentile(slopes, [2.5, 97.5])
            y_lower = model.intercept_ + lower * x_fit
            y_upper = model.intercept_ + upper * x_fit
            plt.fill_between(x_fit, y_lower, y_upper, color='red', alpha=0.2, label='95% CI (slope)')

        # Kendall's tau
        tau, p_tau = kendalltau(x, y)
        plt.annotate(f"Kendall’s τ = {tau:.2f} (p={p_tau:.3g})", (0.02, 0.92), xycoords='axes fraction', fontsize=12)

        plt.xlabel("Dataset Demographic Parity (Unfairness)")
        plt.ylabel(f"Base Model {name} (Difference)")
        plt.title(f"Correlation: Dataset vs. Model {name}")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fairness_correlation_kendall_theilsen.png"))
    plt.close()
