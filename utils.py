import os
import torch
import sklearn
import pandas as pd
import scanpy as sc
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from scipy.sparse import issparse
from sklearn.neighbors import kneighbors_graph



def load_data(args):
    if args.dataset == "DLPFC":
        input_dir = os.path.join('Data\\', args.dataset, args.slice)
        adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        label = pd.read_csv(os.path.join('Data\\', args.dataset, args.slice + '/metadata.tsv'), sep='\t')
        adata.obs['Ground Truth'] = label['layer_guess'].values
        adata.var_names_make_unique()
    elif args.dataset == "Human_breast_cancer(10x)":
        input_dir = os.path.join('D:\Data\\', args.dataset)
        adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        label = pd.read_csv(os.path.join('D:\Data\\', args.dataset + '/metadata.tsv'), sep='\t')
        adata.obs['Ground Truth'] = label['fine_annot_type'].values
        adata.var_names_make_unique()
    elif args.dataset == "Mouse_visual_cortex(STARmap)":
        input_dir = os.path.join('D:\Data\\', args.dataset)
        adata = sc.read(input_dir + '/STARmap_20180505_BY3_1k.h5ad')
        adata.obs['Ground Truth'] = adata.obs['label']
    else:
        input_dir = os.path.join('D:\Data\\', args.dataset)
        adata = sc.read(input_dir + '/E10.5_E1S1.MOSTA.h5ad')
        adata.obs['Ground Truth'] = adata.obs['annotation']

    prefilter_genes(adata, min_cells=3)
    adata = adata[~pd.isnull(adata.obs['Ground Truth'])]
    labels = adata.obs['Ground Truth']

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    pca = PCA(n_components=args.npca)
    if issparse(adata.X):
        pca.fit(adata.X.A)
        embed = pca.transform(adata.X.A)
    else:
        pca.fit(adata.X)
        embed = pca.transform(adata.X)
    features = torch.FloatTensor(np.array(embed))

    return adata, features, labels


def load_Stereo_data(args):
    input_dir = os.path.join('D:/Data/', args.dataset + '/Stero-seq' + '/filtered_feature_bc_matrix.h5ad')
    adata = sc.read_h5ad(input_dir)
    adata.var_names_make_unique()

    adata.var_names_make_unique()
    prefilter_genes(adata, min_cells=3)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    pca = PCA(n_components=args.npca)
    if issparse(adata.X):
        pca.fit(adata.X.A)
        embed = pca.transform(adata.X.A)
    else:
        pca.fit(adata.X)
        embed = pca.transform(adata.X)
    features = torch.FloatTensor(np.array(embed))

    return adata, features


def prefilter_genes(adata, min_counts=None, max_counts=None, min_cells=10, max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp = np.asarray([True] * adata.shape[1], dtype=bool)
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_cells=min_cells)[0]) if min_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_cells=max_cells)[0]) if max_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_counts=min_counts)[0]) if min_counts is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_counts=max_counts)[0]) if max_counts is not None else id_tmp
    adata._inplace_subset_var(id_tmp)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def features_construct_graph(features, k=15, mode="connectivity", metric="cosine"):
    features = features.cpu().numpy()
    adj_ori = kneighbors_graph(features, k, mode=mode, metric=metric, include_self=True)
    adj_ori = adj_ori.toarray()
    row, col = np.diag_indices_from(adj_ori)
    adj_ori[row, col] = 0
    adj_ori = torch.FloatTensor(adj_ori)

    adj = sp.coo_matrix(adj_ori, dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    adj_label = adj_ori + torch.eye(adj_ori.shape[0])

    pos_weight = (adj_ori.shape[0] ** 2 - adj_ori.sum()) / adj_ori.sum()
    norm = adj_ori.shape[0] ** 2 / (2 * (adj_ori.shape[0] ** 2 - adj_ori.sum()))

    return adj, adj_ori, adj_label, pos_weight, norm


def spatial_construct_graph(adata, radius=150):
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    adj_ori = np.zeros((coor.shape[0], coor.shape[0]))
    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)  # (4221, 1)
    for it in range(indices.shape[0]):
        adj_ori[[it] * indices[it].shape[0], indices[it]] = 1
    row, col = np.diag_indices_from(adj_ori)
    adj_ori[row, col] = 0
    adj_ori = torch.FloatTensor(adj_ori)

    adj = sp.coo_matrix(adj_ori, dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    adj_label = adj_ori + torch.eye(adj_ori.shape[0])

    pos_weight = (adj_ori.shape[0] ** 2 - adj_ori.sum()) / adj_ori.sum()
    norm = adj_ori.shape[0] ** 2 / (2 * (adj_ori.shape[0] ** 2 - adj_ori.sum()))

    return adj, adj_ori, adj_label, pos_weight, norm


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def Reconstruct_Ratio(pred, adj_ori):
    adj_pred = pred.reshape(-1)
    adj_pred = (sigmoid(adj_pred) > 0.5).float()
    adj_true = (adj_ori + torch.eye(adj_ori.shape[0])).reshape(-1)
    adj_acc = float(adj_pred.eq(adj_true).sum().item()) / adj_pred.shape[0]
    return adj_acc


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STMIGCL', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata
