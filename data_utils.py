import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config import DATA_PATH, SEED

def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Age_protected'] = (df['Age'] == '>35').astype(int)
    return df

def preprocess_features(df, target, protected, features):
    X = df[features].copy()
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    y = LabelEncoder().fit_transform(df[target])
    protected_attr = df[protected].values
    return X, y, protected_attr

def bias_data(df, protected, target, bias_level):
    df_biased = df.copy()
    mask = (df_biased[protected] == 1)
    employed_mask = (df_biased[target] == df_biased[target].unique()[1])
    n_total = (mask & employed_mask).sum()
    n_flip = int(bias_level * n_total)
    if n_flip > 0:
        affected_idx = df_biased[mask & employed_mask].sample(n=n_flip, random_state=SEED).index
        df_biased.loc[affected_idx, target] = df_biased[target].unique()[0]
    return df_biased