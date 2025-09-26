import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, ttest_rel

from config import set_seed, RESULTS_DIR, BIAS_LEVELS
from data_utils import load_data, preprocess_features, bias_data
from train_baseline import train_baseline
from train_adversarial import train_adversarial
from metrics import fairness_metrics, dataset_demographic_parity, efficiency_metrics
from plot_utils import plot_fairness, plot_efficiency, plot_weights
from plot_utils import plot_fairness_correlation_kendall_theilsen

def main():
    set_seed()
    df = load_data()
    target = 'Employed'
    protected = 'Age_protected'
    features = [col for col in df.columns if col not in [target, protected, 'Age']]

    results_baseline, results_adv = [], []
    eff_baseline, eff_adv = [], []

    for idx, bias in enumerate(BIAS_LEVELS):
        # Step 1: Bias dataset
        df_biased = bias_data(df, protected, target, bias)
        X, y, protected_attr = preprocess_features(df_biased, target, protected, features)
        dp_dataset = dataset_demographic_parity(y, protected_attr)
        X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
            X, y, protected_attr, test_size=0.3, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Baseline
        base_model, base_pred, base_pred_proba = train_baseline(X_train_scaled, y_train, X_test_scaled)
        # Adversarial
        adv_model, adv_pred, adv_pred_proba = train_adversarial(X_train_scaled, y_train, prot_train, X_test_scaled)

        # Save weights for statistical proof
        if idx == 0:
            weights_baseline = base_model.net[0].weight.data.cpu().numpy().flatten()
            weights_adv = adv_model.net[0].weight.data.cpu().numpy().flatten()
            corr, corr_p = pearsonr(weights_baseline, weights_adv)
            _, t_p = ttest_rel(weights_baseline, weights_adv)

        # Fairness
        dp_base, eo_base, eod_base = fairness_metrics(y_test, base_pred, prot_test)
        dp_adv, eo_adv, eod_adv = fairness_metrics(y_test, adv_pred, prot_test)
        results_baseline.append({"dataset_dp": dp_dataset, "model_dp": dp_base, "model_eo": eo_base, "model_eod": eod_base})
        results_adv.append({"dataset_dp": dp_dataset, "model_dp": dp_adv, "model_eo": eo_adv, "model_eod": eod_adv})

        # Efficiency
        eff_baseline.append(efficiency_metrics(y_test, base_pred, base_pred_proba))
        eff_adv.append(efficiency_metrics(y_test, adv_pred, adv_pred_proba))

    # Save and plot
    res_base = pd.DataFrame(results_baseline)
    res_adv = pd.DataFrame(results_adv)
    eff_base_df = pd.DataFrame(eff_baseline)
    eff_adv_df = pd.DataFrame(eff_adv)
    res_base.to_csv(os.path.join(RESULTS_DIR, "baseline_fairness.csv"), index=False)
    res_adv.to_csv(os.path.join(RESULTS_DIR, "adversarial_fairness.csv"), index=False)
    eff_base_df.to_csv(os.path.join(RESULTS_DIR, "baseline_efficiency.csv"), index=False)
    eff_adv_df.to_csv(os.path.join(RESULTS_DIR, "adversarial_efficiency.csv"), index=False)
    # Plots
    plot_fairness(res_base, res_adv, RESULTS_DIR)
    plot_efficiency(res_base, eff_base_df, res_adv, eff_adv_df, RESULTS_DIR)
    plot_weights(weights_baseline, weights_adv, corr, corr_p, t_p, RESULTS_DIR)
    plot_fairness_correlation_kendall_theilsen(res_base, RESULTS_DIR)
    print(f"Saved all experiment results and plots to {RESULTS_DIR}")

if __name__ == "__main__":
    main()