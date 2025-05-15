import json
from typing import Any


def parse_config(file_path: str):
    """
    Parse a configuration file and return the configuration parameters.

    Parameters
    ----------
    file_path : str
        The path to the configuration file.

    Returns
    -------
    tuple
        A tuple of the configuration parameters in the following order:
        generate_data, n_graphs, n_nodes, batch_size, learning_rate, epochs, ml_model, pqc_layers,
        variational_ansatz, use_encoding_param
    """

    with open(file_path) as file:
        config: dict[str, Any] = json.load(file)
        generate_data: bool = config.get("generate_data", False)
        n_graphs: int = config.get("n_graphs", 200)
        n_nodes: int = config.get("n_nodes", 5)
        batch_size: int = config.get("batch_size", 16)
        learning_rate: float = config.get("learning_rate", 0.1)
        epochs: int = config.get("epochs", 20)
        ml_model: str = config.get("ml_model", "quantum")
        pqc_layers: int = config.get("pqc_layers", 3)
        variational_ansatz: str = config.get("variational_ansatz", "rx")
        use_encoding_param: bool = config.get("use_encoding_param", False)
        return (
            generate_data,
            n_graphs,
            n_nodes,
            batch_size,
            learning_rate,
            epochs,
            ml_model,
            pqc_layers,
            variational_ansatz,
            use_encoding_param,
        )
