# STMIGCL：Exploring the latent information in spatial transcriptomics data via multi-view graph convolutional network based on implicit contrastive learning
## Overview
STMIGCL first construct multiple neighbor graphs based on gene expression profiles and spatial location information, and each neighbor graph represents a specific graph structure. It obtain latent representations for each view by utilizing Graph Convolutional Networks (GCNs). Simultaneously, for each neighborhood graph, STMIGCL train a Variational Graph Autoencoder (VGAE) aimed at reconstructing the graph topology. STMIGCL leverage the enhancements from the learned latent space in VGAE as the source of contrast. Through graph contrastive learning with implicit contrastive loss, It further refine the latent representations of each view. To capture the importance of different neighbor graphs, STMIGCL further use an attention mechanism to adaptively merge the representations from different views, obtaining the final spot representations.
!(STMIGCL.png)
# Package: STMIGCL
## Installation
```
git clone git@github.com:YuBinLab-QUST/STMIGCL.git
cd STMIGCL
```
## Usage
