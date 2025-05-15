import torch
from torch.utils.data import TensorDataset, DataLoader

import utils, plots
from models.quantum_model import create_qnode, train_quantum_model
from data.data_gen import generate_graph_data, load_data, save_data

(
    generate_data,
    n_graphs,
    n_nodes,
    batch_size,
    n_layers,
    learning_rate,
    epochs,
    variational_ansatz,
    use_encoding_param,
) = utils.parse_config("config.json")

# Data
if generate_data:
    graphs, labels = generate_graph_data(n_graphs, n_nodes)
    save_data(graphs, labels, n_nodes)
else:
    graphs, labels, load_n_nodes = load_data()
    if graphs.shape[0] != n_graphs or labels.shape[0] != n_graphs:
        raise ValueError(
            f"Number of graphs in saved data ({graphs.shape[0]}) does not match config input n_graphs ({n_graphs})."
        )
    if n_nodes != load_n_nodes:
        raise ValueError(
            f"Number of nodes in saved data ({load_n_nodes}) does not match config input n_nodes ({n_nodes})."
        )

X = torch.tensor(graphs, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Parameters
if variational_ansatz == "rx":
    thetas = torch.nn.Parameter(torch.randn(n_layers, requires_grad=True))
elif variational_ansatz == "rx_ry":
    thetas = torch.nn.Parameter(torch.randn(n_layers, 2, requires_grad=True))
elif variational_ansatz == "rx_ry_rz":
    thetas = torch.nn.Parameter(torch.randn(n_layers, 3, requires_grad=True))
else:
    raise ValueError(f"Invalid variational ansatz: {variational_ansatz}")

if use_encoding_param:
    gammas = torch.nn.Parameter(torch.ones(n_layers, requires_grad=True))
else:
    gammas = None

# Model setup
qnode = create_qnode(
    n_nodes,
    depth=n_layers,
    variational_ansatz=variational_ansatz,
    use_encoding_param=use_encoding_param,
)

train_quantum_model(qnode, thetas, gammas, learning_rate, epochs, loader)

# plots.plot_circuit(graphs, thetas, gammas, qnode, use_encoding_param)
