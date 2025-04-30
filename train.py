import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models.quantum_model import create_qnode
from utils import generate_graph_data

# Config
n_graphs = 200
n_nodes = 4
batch_size = 16
n_layers = 3
learning_rate = 0.1
epochs = 20
variational_ansatz = "rx_ry"  # or "rx"
use_param_encoding = False

# Data
graphs, labels = generate_graph_data(n_graphs, n_nodes)
X = torch.tensor(graphs, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model setup
qnode = create_qnode(n_nodes, L=n_layers, variational_ansatz=variational_ansatz, use_param_encoding=use_param_encoding)

# Parameters
if variational_ansatz == "rx":
    thetas = torch.nn.Parameter(torch.randn(n_layers, requires_grad=True))
elif variational_ansatz == "rx_ry":
    thetas = torch.nn.Parameter(torch.randn(n_layers, 2, requires_grad=True))

if use_param_encoding:
    gammas = torch.nn.Parameter(torch.ones(n_layers, requires_grad=True))
else:
    gammas = None

optimizer = torch.optim.Adam([thetas] + ([gammas] if gammas is not None else []), lr=learning_rate)
loss_fn = nn.BCELoss()

# Training loop
for epoch in range(epochs):
    total_loss = 0
    for xb, yb in loader:
        preds = []
        for i in range(len(xb)):
            out = qnode(xb[i].detach().numpy(), thetas, gammas)
            preds.append(out)

        preds = torch.sigmoid(torch.stack(preds).squeeze())
        loss = loss_fn(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1:02d}: Loss = {total_loss/len(loader):.4f}")
