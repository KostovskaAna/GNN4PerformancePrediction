# Geometric Learning in Black-Box Optimization

This repository contains the code accompanying the paper:

**"Geometric Learning in Black-Box Optimization: A GNN Framework for Algorithm Performance Prediction"**

In this work, we propose a heterogeneous graph neural network (GNN) framework for predicting the performance of modular optimization algorithms. By modeling the relationships between problems, algorithms, and their configurations using a graph-based representation, we demonstrate significant improvements over traditional tabular learning methods.

## 📂 Repository Structure

- `main.py`: The main script to train and evaluate the GNN model.
- `GNN_architecture.py`: Defines the heterogeneous GNN model used for performance prediction.
- `utils.py`: Contains utility functions for graph construction, data handling, and training routines.
- `data/`: Directory containing the processed datasets and metadata required for training.
- `GNN_receptive_field/`: Scripts and utilities for analyzing the impact of the GNN’s receptive field (e.g., varying number of layers).

## 🔧 Setup and Requirements

The code is implemented using [DGL](https://www.dgl.ai/) and PyTorch. For installation, see the official DGL installation guide depending on your platform and CUDA version.

Basic dependencies:
- `dgl`
- `torch`
- `numpy`
- `scikit-learn`

## 💡 Citation

If you use this code or find our work useful, please cite our paper (citation info coming soon).
