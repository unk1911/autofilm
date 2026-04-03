import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from .config import DATA_DIR

MODEL_FILE = DATA_DIR / 'recommendation_model.pt'


class RecommendationNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 100,
    lr: float = 0.001,
    model_file: Path = DATA_DIR / 'recommendation_model.pt',
):
    """Train the neural network on user ratings."""
    print(f"Training on {len(X_train)} films...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    X_tensor = torch.from_numpy(X_train).to(device)
    y_tensor = torch.from_numpy(y_train).reshape(-1, 1).to(device)

    model = RecommendationNetwork(X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = loss_fn(pred, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch+1:3d}: loss={loss.item():.6f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  early stopping at epoch {epoch+1}")
                break

    torch.save(model.state_dict(), model_file)
    print(f"Model saved → {model_file}")
    return model


def load_model(
    input_dim: int,
    model_file: Path = DATA_DIR / 'recommendation_model.pt',
) -> RecommendationNetwork:
    """Load pre-trained model."""
    model = RecommendationNetwork(input_dim)
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    return model.eval()


def predict(model, X_all: np.ndarray, batch_size: int = 1000) -> np.ndarray:
    """Run inference on all movies."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    predictions = []
    for i in range(0, len(X_all), batch_size):
        batch = torch.from_numpy(X_all[i : i + batch_size]).to(device)
        with torch.no_grad():
            pred = model(batch)
        predictions.append(pred.cpu().numpy())

    return np.concatenate(predictions).flatten() * 10.0  # Scale back to 1-10
