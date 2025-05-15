import json
import itertools
from typing import Any
import os

from data.data_gen import generate_graph_data, load_data, save_data


def parse_config(file_path: str):
    """
    Parse a configuration file that may contain lists of values
    and return a list of all config combinations.

    Parameters
    ----------
    file_path : str
        The path to the configuration file.

    Returns
    -------
    list[dict[str, Any]]
        List of configuration dictionaries representing all combinations.
    """
    with open(file_path) as file:
        raw_config: dict[str, Any] = json.load(file)

    # Set defaults
    defaults = {
        "generate_data": [False],
        "n_graphs": [200],
        "n_nodes": [5],
        "batch_size": [16],
        "learning_rate": [0.1],
        "n_layers": [3],
        "epochs": [20],
        "ml_model": ["quantum"],
        "variational_ansatz": ["rx"],
        "use_encoding_param": [False],
    }

    # Ensure all values are in list form
    for key, default_list in defaults.items():
        if key not in raw_config:
            raw_config[key] = default_list
        elif not isinstance(raw_config[key], list):
            raw_config[key] = [raw_config[key]]

    # Generate all combinations
    keys = list(defaults.keys())
    value_combinations = list(itertools.product(*(raw_config[k] for k in keys)))

    configs = [dict(zip(keys, values)) for values in value_combinations]
    return configs


def get_data_filename(n_graphs, n_nodes):
    """Generate a consistent filename for dataset based on parameters"""
    return f"data/graph_data_{n_graphs}_{n_nodes}.npz"


def generate_or_load_data(n_graphs, n_nodes, generate_data):
    """
    Generate or load graph data based on the specified parameters.

    Parameters
    ----------
    n_graphs : int
        Number of graphs in the dataset.
    n_nodes : int
        Number of nodes in each graph.
    generate_data : bool
        Whether to generate new data or load existing data.

    Returns
    -------
    graphs : np.ndarray
        Array of adjacency matrices representing the graphs.
    labels : np.ndarray
        Array of labels indicating the connectivity of each graph.

    Raises
    ------
    ValueError
        If the loaded data does not match the expected number of graphs or nodes.
    """

    filename = get_data_filename(n_graphs, n_nodes)

    # Check if we need to generate new data
    if generate_data or not os.path.exists(filename):
        print(f"Generating new data for {n_graphs} graphs with {n_nodes} nodes...")
        graphs, labels = generate_graph_data(n_graphs, n_nodes)
        save_data(graphs, labels, n_nodes, filename)
    else:
        print(f"Loading existing data from {filename}...")
        graphs, labels, load_n_nodes = load_data(filename)

        # Verify data matches expected parameters
        if graphs.shape[0] != n_graphs:
            raise ValueError(
                f"Saved graphs ({graphs.shape[0]}) do not match config n_graphs ({n_graphs})"
            )
        if n_nodes != load_n_nodes:
            raise ValueError(
                f"Saved node count ({load_n_nodes}) does not match config n_nodes ({n_nodes})"
            )

    return graphs, labels
