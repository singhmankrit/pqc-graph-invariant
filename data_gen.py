import networkx as nx
import numpy as np

def generate_graph_data(n_graphs=100, n_nodes=4, p_edge=0.5, seed=42):
    np.random.seed(seed)
    graphs, labels = [], []

    for _ in range(n_graphs):
        G = nx.erdos_renyi_graph(n_nodes, p_edge)
        A = nx.to_numpy_array(G)
        label = 0 if nx.is_connected(G) else 1
        graphs.append(A)
        labels.append(label)

    return np.array(graphs), np.array(labels)

def save_data(graphs, labels, filename='data/graph_data.npz'):
    np.savez_compressed(filename, graphs=graphs, labels=labels)

def load_data(filename='data/graph_data.npz'):
    data = np.load(filename)
    return data['graphs'], data['labels']

graphs, labels =generate_graph_data()
save_data(graphs, labels)