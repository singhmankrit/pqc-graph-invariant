import torch
import torch.nn as nn
import torch.optim as optim
from itertools import combinations_with_replacement
from torch.utils.data import DataLoader, TensorDataset


class PolynomialModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze()


def polynomial_features(X, degree):
    """
    X: (N, D) tensor where each row is a flattened adjacency matrix
    degree: the degree of the polynomial (k)

    Returns:
        (N, num_features) tensor with polynomial features
    """
    N, D = X.shape
    combos = list(combinations_with_replacement(range(D), degree))  # All i1...ik
    features = torch.ones((N, len(combos)), dtype=torch.float32)

    for idx, combo in enumerate(combos):
        for i in combo:
            features[:, idx] *= X[:, i]
    return features

def train_polynomial_model(X, y, degree=2, epochs=100, lr=0.01, batch_size=32):
    X_poly = polynomial_features(X, degree)
    dataset = TensorDataset(X_poly, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PolynomialModel(X_poly.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        for xb, yb in loader:
            preds = model(xb)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += ((preds > 0.5).float() == yb).sum().item()

        acc = correct / len(y)
        print(f"Epoch {epoch+1:02d}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

    return model
