import networkx as nx
import pennylane as qml
import matplotlib.pyplot as plt


def plot_loss_accuracy_comparison(train_loss, train_acc, test_loss, test_acc, config):
    """
    Plot training and testing loss & accuracy, with a config box.

    Parameters
    ----------
    train_loss : list
        Training loss per epoch
    train_acc : list
        Training accuracy per epoch
    test_loss : list
        Test loss per epoch
    test_acc : list
        Test accuracy per epoch
    config : dict
        Dictionary of configuration values (e.g., n_nodes, learning_rate, etc.)

    Returns
    -------
    None
    """

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss
    axs[0].plot(train_loss, label="Train Loss", color="red")
    axs[0].plot(test_loss, label="Test Loss", color="green", linestyle="--")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Loss")
    axs[0].grid(True)
    axs[0].legend()

    # Plot Accuracy
    axs[1].plot(train_acc, label="Train Accuracy", color="red")
    axs[1].plot(test_acc, label="Test Accuracy", color="green", linestyle="--")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_ylim(0, 1)
    axs[1].set_title("Accuracy")
    axs[1].grid(True)
    axs[1].legend()

    # Readable config block
    readable_config = [
        "Training Configuration",
        "─" * 24,
        f"Graphs         : {config.get('n_graphs')}",
        f"Nodes          : {config.get('n_nodes')}",
        f"Batch Size     : {config.get('batch_size')}",
        f"Learning Rate  : {config.get('learning_rate')}",
        f"Epochs         : {config.get('epochs')}",
        f"Model          : {config.get('ml_model').capitalize()}",
    ]

    # Add model-specific configs
    if config.get("ml_model") == "quantum":
        readable_config += [
            f"Layers         : {config.get('n_layers')}",
            f"Ansatz         : {config.get('variational_ansatz').upper()}",
            f"Encoding Param : {config.get('use_encoding_param')}",
        ]
    else:
        readable_config += [
            f"Degree         : {config.get('n_layers')}",
        ]

    config_text = "\n".join(readable_config)

    # Plot config box on the side
    fig.text(
        0.98,
        0.5,
        config_text,
        fontsize=10,
        va="center",
        ha="left",
        family="monospace",
        bbox=dict(
            facecolor="whitesmoke",
            edgecolor="gray",
            boxstyle="round,pad=1.2",
            alpha=0.95,
        ),
    )

    # Construct plot name safely and uniquely
    model = config.get("ml_model")
    base = f"{model}_nodes{config.get('n_nodes')}_layers{config.get('n_layers')}_epochs{config.get('epochs')}"

    if model == "quantum":
        extra = f"_ansatz_{config.get('variational_ansatz')}_param_{config.get('use_encoding_param')}"
    else:
        extra = ""

    plot_name = f"images/{base}{extra}.png"

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(plot_name, bbox_inches="tight")
    plt.close()


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
