import torch
import utils, plots
from models.classical_model import train_polynomial_model
from models.quantum_model import create_qnode, train_quantum_model
from data.data_gen import generate_graph_data, load_data, save_data

all_configs = utils.parse_config("config.json")

for config in all_configs:
    print(f"\nRunning with config: {config}")

    # Unpack config values
    generate_data = config["generate_data"]
    n_graphs = config["n_graphs"]
    n_nodes = config["n_nodes"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    n_layers = config["n_layers"]
    epochs = config["epochs"]
    ml_model = config["ml_model"]
    variational_ansatz = config.get("variational_ansatz")
    use_encoding_param = config.get("use_encoding_param")

    # === SKIP incompatible parameter combinations ===
    if ml_model == "classical" and (
        variational_ansatz is not None or use_encoding_param is not None
    ):
        print(
            f"⚠️ Skipping config: classical model should not specify quantum-only parameters "
            f"(variational_ansatz: {variational_ansatz}, use_encoding_param: {use_encoding_param})"
        )
        continue

    if ml_model == "quantum" and (
        variational_ansatz is None or use_encoding_param is None
    ):
        print(
            f"⚠️ Skipping config: quantum model requires 'variational_ansatz' and 'use_encoding_param'."
        )
        continue

    # Data
    if generate_data:
        graphs, labels = generate_graph_data(n_graphs, n_nodes)
        save_data(graphs, labels, n_nodes)
    else:
        graphs, labels, load_n_nodes = load_data()
        if graphs.shape[0] != n_graphs or labels.shape[0] != n_graphs:
            raise ValueError(
                f"Saved graphs ({graphs.shape[0]}) do not match config n_graphs ({n_graphs})"
            )
        if n_nodes != load_n_nodes:
            raise ValueError(
                f"Saved node count ({load_n_nodes}) does not match config n_nodes ({n_nodes})"
            )

    if ml_model == "quantum":
        X = torch.tensor(graphs, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)

        if variational_ansatz == "rx":
            thetas = torch.nn.Parameter(torch.randn(n_layers, requires_grad=True))
        elif variational_ansatz == "rx_ry":
            thetas = torch.nn.Parameter(torch.randn(n_layers, 2, requires_grad=True))
        elif variational_ansatz == "rx_ry_rz":
            thetas = torch.nn.Parameter(torch.randn(n_layers, 3, requires_grad=True))
        else:
            raise ValueError(f"Invalid variational ansatz: {variational_ansatz}")

        gammas = (
            torch.nn.Parameter(torch.ones(n_layers, requires_grad=True))
            if use_encoding_param
            else None
        )

        qnode = create_qnode(
            n_nodes,
            depth=n_layers,
            variational_ansatz=variational_ansatz,
            use_encoding_param=use_encoding_param,
        )

        train_quantum_model(
            X, y, qnode, thetas, gammas, learning_rate, epochs, batch_size, config
        )

    elif ml_model == "classical":
        X = torch.tensor(graphs, dtype=torch.float32).reshape(len(graphs), -1)
        y = torch.tensor(labels, dtype=torch.float32)

        model = train_polynomial_model(
            X,
            y,
            degree=n_layers,
            epochs=epochs,
            lr=learning_rate,
            batch_size=batch_size,
            config=config,
        )

    else:
        raise ValueError(f"Invalid ML model: {ml_model}")
