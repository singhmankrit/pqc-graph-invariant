import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from models.quantum_model import create_qnode
from data_gen import generate_graph_data

# Config
n_graphs = 200
n_nodes = 4
batch_size = 16
n_layers = 3
learning_rate = 0.1
epochs = 20
variational_ansatz = "rx"  # or "rx_ry"
use_encoding_param = False

# Data
graphs, labels = generate_graph_data(n_graphs, n_nodes)
X = torch.tensor(graphs, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model setup
qnode = create_qnode(
    n_nodes,
    depth=n_layers,
    variational_ansatz=variational_ansatz,
    use_encoding_param=use_encoding_param,
)

# Parameters
if variational_ansatz == "rx":
    thetas = torch.nn.Parameter(torch.randn(n_layers, requires_grad=True))
elif variational_ansatz == "rx_ry":
    thetas = torch.nn.Parameter(torch.randn(n_layers, 2, requires_grad=True))

if use_encoding_param:
    gammas = torch.nn.Parameter(torch.ones(n_layers, requires_grad=True))
else:
    gammas = None

optimizer = torch.optim.Adam(
    [thetas] + ([gammas] if gammas is not None else []), lr=learning_rate
)
loss_fn = nn.BCELoss()

loss_list = []
acc_list = []

for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    for xb, yb in loader:
        preds = []
        for i in range(len(xb)):
            out = qnode(xb[i].detach().numpy(), thetas, gammas)
            preds.append(out)

        preds = torch.sigmoid(torch.stack(preds).squeeze()).float()
        loss = loss_fn(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted_labels = (preds > 0.5).float()
        correct += (predicted_labels == yb).sum().item()
        total += len(yb)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    loss_list.append(avg_loss)
    acc_list.append(accuracy)
    print(f"Epoch {epoch+1:02d}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")

# Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_list, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(acc_list, label="Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)

plt.tight_layout()
plt.show()
