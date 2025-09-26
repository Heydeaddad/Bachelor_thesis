import torch
from models import MLP
from config import BASELINE_EPOCHS, BASELINE_LR

def train_baseline(X_train, y_train, X_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    model = MLP(X_train.shape[1]).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=BASELINE_LR)
    for epoch in range(BASELINE_EPOCHS):
        model.train()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        y_pred_proba = model(X_test_tensor).squeeze().cpu().numpy()
        y_pred = (y_pred_proba >= 0.5).astype(int)
    return model, y_pred, y_pred_proba