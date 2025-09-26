import os
import torch
import numpy as np
import random

# Paths
DATA_PATH = "stackoverflow_full.csv"
RESULTS_DIR = "experiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Random seeds for reproducibility
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Training params
BASELINE_EPOCHS = 300
ADVERSARIAL_EPOCHS = 300
BASELINE_LR = 0.005
ADVERSARIAL_LR = 0.005
ALPHA = 1.0  # Adversarial loss weight
BIAS_LEVELS = np.linspace(0, 0.8, 30)