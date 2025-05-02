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
        n_graphs, n_nodes, batch_size, n_layers, learning_rate, epochs,
        variational_ansatz, use_encoding_param
    """

    with open(file_path) as file:
        config: dict[str, Any] = json.load(file)
        n_graphs: int = config.get("n_graphs", 200)
        n_nodes: int = config.get("n_nodes", 5)
        batch_size: int = config.get("batch_size", 16)
        n_layers: int = config.get("n_layers", 3)
        learning_rate: float = config.get("learning_rate", 0.1)
        epochs: int = config.get("epochs", 20)
        variational_ansatz: str = config.get("variational_ansatz", "rx")
        use_encoding_param: bool = config.get("use_encoding_param", False)
        return (
            n_graphs,
            n_nodes,
            batch_size,
            n_layers,
            learning_rate,
            epochs,
            variational_ansatz,
            use_encoding_param,
        )
