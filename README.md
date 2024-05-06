# STMIGCL：Exploring the latent information in spatial transcriptomics data via multi-view graph convolutional network based on implicit contrastive learning
## Overview
STMIGCL first construct multiple neighbor graphs based on gene expression profiles and spatial location information, and each neighbor graph represents a specific graph structure. It obtain latent representations for each view by utilizing Graph Convolutional Networks (GCNs). Simultaneously, for each neighborhood graph, STMIGCL train a Variational Graph Autoencoder (VGAE) aimed at reconstructing the graph topology. STMIGCL leverage the enhancements from the learned latent space in VGAE as the source of contrast. Through graph contrastive learning with implicit contrastive loss, It further refine the latent representations of each view. To capture the importance of different neighbor graphs, STMIGCL further use an attention mechanism to adaptively merge the representations from different views, obtaining the final spot representations.
<img src="https://github.com/YuBinLab-QUST/STMIGCL/blob/main/STMIGCL.png">
# Package: STMIGCL
## Installation
```
git clone git@github.com:YuBinLab-QUST/STMIGCL.git
cd STMIGCL
```
## Usage
example DLPFC dataset
```
import scanpy as sc
from STMIGCL.train import train
from args import init_args
from utils import load_data

import matplotlib.pyplot as plt

args = init_args()
adata, features, _ = load_data(args)

y_pred = train(args, adata, features)
adata.obs['pred'] = y_pred
adata.obs["pred"] = adata.obs["pred"].astype('category')

sc.pl.spatial(adata,
              img_key="hires",
              color=["pred"],
              title=["STMIGCL"],
              show=True)
```
# Requirements
python 3.9 <br> 
numpy <br>
scikit-learn <br>
scanpy <br>
torch <br>
networkx <br>
# Dataset
(1) the DLPFC dataset is accessible within the spatialLIBD package http://spatial.libd.org/spatialLIBD <br>
(2) 10x Visium spatial transcriptomics dataset of human breast cancer https://github.com/JinmiaoChenLab/SEDR_analyses/tree/master/data <br>
(3) the processed Stereo-seq data from mouse olfactory bulb tissue is accessible on https://github.com/JinmiaoChenLab/SEDR_analyses <br>
(4) the Stereo-seq data acquired from mouse embryos at E9.5 can be downloaded from https://db.cngb.org/stomics/mosta/ <br>
(5) the mouse visual cortex STARmap data is accessible on https://www.starmapresources.com/data <br>
