import networkx as nx
import pennylane as qml
import matplotlib.pyplot as plt


def plot_loss_accuracy(loss_list, acc_list):
    """
    Plot the loss and accuracy of the model during training.

    Parameters
    ----------
    loss_list : list
        A list of loss values at each epoch.
    acc_list : list
        A list of accuracy values at each epoch.

    Returns
    -------
    None
    """

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_list, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(acc_list, label="Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("plots.png")


def plot_circuit(graphs, thetas, gammas, qnode, use_encoding_param=False):
    """
    Visualize a quantum circuit and save the plot as an image.
    Sanity check to see the circuit.

    Parameters
    ----------
    graphs : np.ndarray
        Array of adjacency matrices representing the graphs.
    thetas : torch.Tensor
        The parameters of the variational ansatz.
    gammas : torch.Tensor
        The parameters of the encoding.
    qnode : function
        The Pennylane QNode representing the quantum circuit.
    use_encoding_param : bool, optional
        Whether to use the encoding parameter in the circuit. Default is False.

    Notes
    -----
    The plot is saved as 'circuit.png'.
    """

    adj_matrix_sample = graphs[0]
    if use_encoding_param:
        gammas_sample = gammas.detach()
    else:
        gammas_sample = None

    qml.draw_mpl(qnode)(adj_matrix_sample, thetas, gammas_sample)
    plt.savefig("circuit.png")


def visually_verify_graphs(graphs, labels, num_to_plot=5):
    """
    Visually verify a set of graphs by plotting them and saving the plots as images.
    Sanity check to see the graphs.

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
    For each graph, a plot is created and saved as an image file named 'graph-{i}.png',
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
        plt.savefig(f"graph-{i}.png")
