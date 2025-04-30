import pennylane as qml
from pennylane import numpy as np

def get_ising_hamiltonian(adj_matrix):
    n = len(adj_matrix)
    H = qml.Hamiltonian([], [])

    for i in range(n):
        for j in range(i+1, n):
            if adj_matrix[i, j] == 1:
                H += qml.Hamiltonian([1.0], [qml.PauliZ(i) @ qml.PauliZ(j)])

    return H

def create_qnode(n_qubits, L=2, variational_ansatz="rx", use_param_encoding=False):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(adj_matrix, thetas, gammas=None):
        H = get_ising_hamiltonian(adj_matrix)

        for l in range(L):
            γ = gammas[l] if use_param_encoding else 1.0
            qml.ApproxTimeEvolution(H, γ, 1)

            for i in range(n_qubits):
                if variational_ansatz == "rx":
                    qml.RX(thetas[l], wires=i)
                elif variational_ansatz == "rx_ry":
                    qml.RX(thetas[l][0], wires=i)
                    qml.RY(thetas[l][1], wires=i)

        return qml.expval(qml.PauliZ(0))

    return circuit
