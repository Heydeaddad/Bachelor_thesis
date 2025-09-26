import torch
from models import MLP, AdvNet
from config import ADVERSARIAL_EPOCHS, ADVERSARIAL_LR, ALPHA

def train_adversarial(X_train, y_train, protected_train, X_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    protected_train_tensor = torch.tensor(protected_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    model = MLP(X_train.shape[1]).to(device)
    adv = AdvNet().to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=ADVERSARIAL_LR)
    optimizer_adv = torch.optim.Adam(adv.parameters(), lr=ADVERSARIAL_LR)
    for epoch in range(ADVERSARIAL_EPOCHS):
        # Step 1: Train adversary
        model.eval()
        adv.train()
        out = model(X_train_tensor).detach()
        p_pred = adv(out)
        loss_adv = criterion(p_pred, protected_train_tensor)
        optimizer_adv.zero_grad()
        loss_adv.backward()
        optimizer_adv.step()
        # Step 2: Train main model
        model.train()
        adv.eval()
        out = model(X_train_tensor)
        y_pred = out
        p_pred = adv(out)
        loss_main = criterion(y_pred, y_train_tensor)
        loss = loss_main - ALPHA * criterion(p_pred, protected_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        y_pred_proba = model(X_test_tensor).squeeze().cpu().numpy()
        y_pred = (y_pred_proba >= 0.5).astype(int)
    return model, y_pred, y_pred_proba