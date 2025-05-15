import torch
import torch.nn as nn
import torch.optim as optim
from itertools import combinations_with_replacement
from torch.utils.data import DataLoader, TensorDataset, random_split

import plots


class PolynomialModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze()


def polynomial_features(X, degree):
    """
    Generate polynomial features of a given degree.

    Args:
        X (tensor): shape (N, D), input features
        degree (int): polynomial degree

    Returns:
        Tensor of shape (N, num_features)
    """
    N, D = X.shape
    combos = list(combinations_with_replacement(range(D), degree))
    features = torch.ones((N, len(combos)), dtype=torch.float32)

    for idx, combo in enumerate(combos):
        for i in combo:
            features[:, idx] *= X[:, i]
    return features


def train_polynomial_model(
    X, y, degree=2, epochs=100, lr=0.1, batch_size=16, config=None
):
    X_poly = polynomial_features(X, degree)

    # --- Split into training and test sets
    dataset = TensorDataset(X_poly, y)
    test_size = int(len(dataset) * 0.2)  # Test Split = 20%
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = PolynomialModel(X_poly.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            preds = model(xb)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            predicted = (preds > 0.5).float()
            correct += (predicted == yb).sum().item()
            total += yb.size(0)

        avg_train_loss = total_train_loss / len(train_loader)

        train_accuracy = correct / total
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(train_accuracy)

        # Evaluation on test set
        model.eval()
        with torch.no_grad():
            total_test_loss = 0
            correct_test = 0
            total_test = 0
            for xb, yb in test_loader:
                preds = model(xb)
                total_test_loss += loss.item()
                predicted = (preds > 0.5).float()
                correct_test += (predicted == yb).sum().item()
                total_test += yb.size(0)

            avg_test_loss = total_test_loss / len(test_loader)
            test_accuracy = correct_test / total_test
            test_loss_list.append(avg_test_loss)
            test_acc_list.append(test_accuracy)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
            f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}"
        )

    plots.plot_loss_accuracy_comparison(
        train_loss_list, train_acc_list, test_loss_list, test_acc_list, config
    )

    return model
