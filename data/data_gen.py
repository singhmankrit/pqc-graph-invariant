import networkx as nx
import numpy as np


def generate_graph_data(n_graphs=100, n_nodes=4, p_edge=0.35, seed=42):
    """
    Generate random graph data with labels for connectivity.

    Parameters
    ----------
    n_graphs : int
        Number of graphs to generate. Default is 100.
    n_nodes : int
        Number of nodes in the graph. Default is 4.
    p_edge : float
        Probability of an edge between two nodes. Default is 0.35 because otherwise the graphs are almost certainly connected for higher number of nodes.
    seed : int
        Random seed for generating graphs. Default is 42.

    Returns
    -------
    graphs : np.ndarray
        Array of shape (n_graphs, n_nodes, n_nodes) containing a list of the adjacency
        matrices of the generated graphs.
    labels : np.ndarray
        Array of shape (n_graphs,) containing 0 if the graph is connected and 1
        if it is not.
    """
    np.random.seed(seed)
    graphs, labels = [], []

    for _ in range(n_graphs):
        G = nx.erdos_renyi_graph(n_nodes, p_edge)
        A = nx.to_numpy_array(G)
        label = 1 if nx.is_connected(G) else 0
        graphs.append(A)
        labels.append(label)

    return np.array(graphs), np.array(labels)


def save_data(graphs, labels, n_nodes, filename="data/graph_data.npz"):
    """
    Save graph data and labels to a compressed .npz file.

    Parameters
    ----------
    graphs : np.ndarray
        Array of adjacency matrices representing the graphs to be saved.
    labels : np.ndarray
        Array of labels indicating the connectivity of each graph.
    n_nodes : int
        Number of nodes in generated graph.
    filename : str, optional
        The path to the file where the data should be saved. Default is 'graph_data.npz'.
    """
    np.savez_compressed(filename, graphs=graphs, labels=labels, n_nodes=n_nodes)


def load_data(filename="data/graph_data.npz"):
    """
    Load graph data and labels from a compressed .npz file.

    Parameters
    ----------
    filename : str, optional
        The path to the file where the data is stored. Default is 'graph_data.npz'.

    Returns
    -------
    graphs : np.ndarray
        Array of adjacency matrices representing the loaded graphs.
    labels : np.ndarray
        Array of labels indicating the connectivity of each loaded graph.
    n_nodes : int
        Number of nodes in loaded graph.
    """
    data = np.load(filename)
    return data["graphs"], data["labels"], data["n_nodes"]
