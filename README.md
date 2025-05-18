# Quantum Graph Connectivity Classifier

This project compares classical and quantum machine learning models for learning the **connectivity property** of graphs. It evaluates expressivity and performance of:

- Classical multivariate polynomial classifiers
- Quantum data re-uploading models using different variational ansätze

The script will:
- Load or generate graph data
- Train both classical and quantum models (depending on config)
- Save performance plots to the images/ directory
- A plot of training and test loss/accuracy
- Annotated configuration box in the corner

## 📁 Project Structure
```bash
├── data/
│ └── data_gen.py # Graph Data Generation code
├── models/
│ ├── classical_model.py # Classical polynomial model
│ └── quantum_model.py # Quantum model with Pennylane
├── utils.py # Config parser and helper functions
├── plots.py # Training plots and circuit visualization
├── main.py # Entry point for training/evaluation
├── config.json # Configuration file (grid of settings)
└── README.md # This file
```

## ⚙️ Configuration

All experiments are controlled through the `config.json` file. Multiple values in a list mean different iterations code being run using each config. The following parameters can be set:

| Option | Default | Description |
| ------ | ------- | ----------- |
| generate_data | `[false]` | Controls whether the graph data is regenerated or reused from a previous run. |
| n_graphs | `[200, 600]` | Number of graphs generated using the Erdős–Rényi model. |
| n_nodes | `[4, 5, 6, 7, 8, 9]` | Number of nodes generated in the graph. |
| batch_size | `[16]` | Batch size for training the model. |
| learning_rate | `[0.1]` | Learning rate for the model. |
| n_layers | `[1, 2, 3, 4, 5]` | For Quantum, this is the number of layers of data encoding and variational circuit. For Classical, this is the degree of the polynomial. |
| epochs | `[20, 50]` | Number of epochs for the model to train. |
| ml_model | `["classical", "quantum"]` | Chooses which model to train on. |
| variational_ansatz | `["rx", "rx_ry", "rx_ry_rz", "rx_ry_rz_ising"]` | (Quantum only) Chooses which ansatz to train on, including the proposed `rx_ry_rz_ising` ansatz. |
| use_encoding_param | `[false, true]` | (Quantum only) Chooses if the graph uploading part of the circuit is parametrized or not. |

