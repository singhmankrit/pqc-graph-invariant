import torch

from models.classical_model import train_polynomial_model
from models.quantum_model import create_qnode, train_quantum_model
from utils import parse_config, generate_or_load_data

config_list = parse_config("config.json")

# Store which Classical Configs run
classical_configs_run = set()

for config in config_list:
    generate_data = config["generate_data"]
    n_graphs = config["n_graphs"]
    n_nodes = config["n_nodes"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    n_layers = config["n_layers"]
    epochs = config["epochs"]
    ml_model = config["ml_model"]

    # Load data
    graphs, labels = generate_or_load_data(n_graphs, n_nodes, generate_data)

    # For classical model, skip if we've already run this configuration
    if ml_model == "classical":
        classical_config_key = (
            n_graphs,
            n_nodes,
            batch_size,
            learning_rate,
            n_layers,
            epochs,
        )

        if classical_config_key in classical_configs_run:
            print(f"Skipping duplicate classical model run with config: {config}")
            continue

        classical_configs_run.add(classical_config_key)

        # Run the classical model
        print(f"\nRunning classical model with config: {config}")
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

    elif ml_model == "quantum":
        variational_ansatz = config.get("variational_ansatz")
        use_encoding_param = config.get("use_encoding_param")

        print(f"\nRunning quantum model with config: {config}")
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
    else:
        raise ValueError(f"Invalid ML model: {ml_model}")
