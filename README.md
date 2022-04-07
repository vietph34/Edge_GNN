# Edge_GNN
Re-implementation of Exploiting [Edge Features in Graph Neural Networks](https://arxiv.org/pdf/1809.02709.pdf)

This repo implemented the EGNN(C)-M (GCN without multi-dimensional edge features).

The main difference from original GCN is the edge features is normalized by _doubly stochastic normalization (utils.py, line 5)_ and _apply edge features after the convolution step (layers.py, line 34)_
| Model      | Results |
| ----------- | ----------- |
| Paper      | 81.80    |
|This version |81.50|

