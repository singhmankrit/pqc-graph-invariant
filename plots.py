import networkx as nx
import matplotlib.pyplot as plt


def visually_verify_graphs(graphs, labels, num_to_plot=5):
    """
    Visually verify a set of graphs by plotting them and saving the plots as images.
    This is simply for sanity check.

    Parameters
    ----------
    graphs : np.ndarray
        Array of adjacency matrices representing the graphs to be plotted.
    labels : np.ndarray
        Array of labels indicating the connectivity of each graph.
    num_to_plot : int, optional
        The number of graphs to plot. Default is 5.

    Notes
    -----
    For each graph, a plot is created and saved as an image file named 'fig-{i}.png',
    where 'i' is the index of the graph.
    """

    for i in range(num_to_plot):
        adj = graphs[i]
        label = labels[i]

        # Create a graph from the adjacency matrix
        G = nx.from_numpy_array(adj)

        plt.figure(figsize=(3, 3))
        nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray")
        print(f"Graph {i}, Label: {label}")
        plt.savefig(f"fig-{i}.png")
