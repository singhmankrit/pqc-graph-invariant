import json
import itertools
from typing import Any


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
