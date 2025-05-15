import pennylane as qml
import torch
import torch.nn as nn

import plots


def create_qnode(n_qubits, depth=2, variational_ansatz="rx", use_encoding_param=False):
    """
    Create a Pennylane QNode that implements the quantum circuit for
    learning graph connectivity.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    depth : int, optional
        Number of layers in the circuit. Default is 2.
    variational_ansatz : str, optional
        The type of variational ansatz to use in the circuit.
        Options are "rx" and "rx_ry". Default is "rx".
    use_encoding_param : bool, optional
        Whether to use the encoding parameter in the Ising Hamiltonian. Default is False.

    Returns
    -------
    qnode : function
        A Pennylane QNode that implements the quantum circuit.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(adj_matrix, thetas, gammas=None):
        """
        This function implements the quantum circuit for learning graph connectivity
        using the Ising Hamiltonian to encode the graph data.

        Parameters
        ----------
        adj_matrix : array
            The adjacency matrix of the graph.
        thetas : array
            The parameters of the variational ansatz.
        gammas : array, optional
            The parameters of the encoding. If not provided, will be set to 1.

        Returns
        -------
        expectation value of PauliZ on the first qubit
        """
        for l in range(depth):
            gamma = gammas[l] if use_encoding_param else 1.0

            # we can directly use IsingZZ for each edge
            # because the individual Hamiltonians in the sum commute with each other
            for i in range(len(adj_matrix)):
                for j in range(i + 1, len(adj_matrix)):
                    if adj_matrix[i, j] == 1:
                        qml.IsingZZ(
                            2 * gamma, wires=[i, j]
                        )  # because 2*gamma will be divided by 2 in the exponent

            for i in range(n_qubits):
                if variational_ansatz == "rx":
                    qml.RX(thetas[l], wires=i)
                elif variational_ansatz == "rx_ry":
                    qml.RX(thetas[l][0], wires=i)
                    qml.RY(thetas[l][1], wires=i)

        operator = qml.PauliZ(0)
        for i in range(1, n_qubits):
            operator = operator @ qml.PauliZ(i)
        return qml.expval(operator)

    return circuit


def train_quantum_model(qnode, thetas, gammas, learning_rate, epochs, loader):
    """
    Train the quantum model on given dataset.

    Parameters
    ----------
    qnode : function
        A Pennylane QNode that implements the quantum circuit.
    thetas : torch.nn.Parameter
        The parameters of the variational ansatz.
    gammas : torch.nn.Parameter
        The parameters of the encoding. If not provided, will be set to 1.
    learning_rate : float
        The learning rate of the optimizer.
    epochs : int
        The number of epochs to train the model.
    loader : torch.utils.data.DataLoader
        A DataLoader containing the dataset.

    Returns
    -------
    None
    """

    optimizer = torch.optim.Adam(
        [thetas] + ([gammas] if gammas is not None else []), lr=learning_rate
    )
    loss_fn = nn.BCELoss()

    loss_list = []
    acc_list = []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for xb, yb in loader:
            preds = []
            for i in range(len(xb)):
                out = qnode(xb[i].detach().numpy(), thetas, gammas)
                preds.append(out)

            preds = torch.sigmoid(torch.stack(preds).squeeze()).float()
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted_labels = (preds > 0.5).float()
            correct += (predicted_labels == yb).sum().item()
            total += len(yb)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        loss_list.append(avg_loss)
        acc_list.append(accuracy)
        print(f"Epoch {epoch+1:02d}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")

    plots.plot_loss_accuracy(loss_list, acc_list)
