# Experiment: Fairness and Effectiveness in HR AI Decision Support

## Project Structure

```
experiment/
│
├── config.py                # Configurations, paths, random seeds
├── data_utils.py            # Data loading, preprocessing, bias manipulation
├── models.py                # Model classes (MLP, AdvNet)
├── train_baseline.py        # Baseline training/eval code
├── train_adversarial.py     # Adversarial mitigation training/eval code
├── metrics.py               # All fairness and efficiency metrics
├── plot_utils.py            # Plotting functions for metrics and proofs
├── experiment_runner.py     # Orchestrates the experiment, runs all steps
└── README.md                # This file
```

## Prerequisites

- Python 3.8+
- Required packages:
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - torch
  - (optional, for Jupyter: `ipykernel`)

Install all dependencies with:
```bash
pip install numpy pandas scikit-learn matplotlib torch
```

## Data

- Place your data CSV (`stackoverflow_full.csv`) in the project root or specify its path in `config.py`.

## How to Run

1. **Set up your environment:**
   - Make sure all dependencies are installed.
   - (Optional) Review or modify `config.py` for paths and parameters.

2. **Run the experiment:**
   ```bash
   python experiment/experiment_runner.py
   ```

3. **Results:**
   - All metrics and plots will be saved in the `experiment_results/` directory.
   - Key result files:
     - `adversarial_fairness_comparison.png`
     - `adversarial_efficiency_comparison.png`
     - `main_model_weights_bias0.png`
     - CSV files with raw metric values

4. **Customize/Extend:**
   - To change model parameters, data paths, or experiment options, edit `config.py`.
   - To add more metrics or models, update `metrics.py` and `models.py`.

## Troubleshooting

- If you get a `ModuleNotFoundError`, make sure you are running from the project root and using the correct Python environment.
- For errors about missing data, check the path in `config.py`.
- For GPU training, ensure you have a compatible CUDA environment.

---

## Contact

For questions or issues, please open an issue or contact the author.