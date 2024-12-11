import torch
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from tqdm import tqdm

from VGAE import GCNModelVAE
from loss import VGAE_Loss, target_distribution, Implicit_Contrastive_Loss, kl_loss
from model import STMGCN
from utils import features_construct_graph, spatial_construct_graph, Reconstruct_Ratio


def train(args, adata):
    features = adata.obsm['features']

    adj1, adj1_ori, adj1_label, pos_weight1, norm1 = features_construct_graph(features, args.k)
    adj1 = adj1.to(args.device)
    adj1_label = adj1_label.to(args.device)

    adj2, adj2_ori, adj2_label, pos_weight2, norm2 = spatial_construct_graph(adata, args.radius)
    adj2 = adj2.to(args.device)
    adj2_label = adj2_label.to(args.device)

    VGAE1 = GCNModelVAE(features.shape[1], args.nemb, 0)
    VGAE2 = GCNModelVAE(features.shape[1], args.nemb, 0)
    model = STMGCN(nfeat=features.shape[1], nemb=args.nemb, nclass=args.n_cluster)

    optimizer_vgae1 = torch.optim.Adam(VGAE1.parameters(), lr=args.lr2, weight_decay=0)
    optimizer_vgae2 = torch.optim.Adam(VGAE2.parameters(), lr=args.lr2, weight_decay=0)
    optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.weight_decay)

    emb1, emb2, emb = model.mgcn(features, adj1, adj2)

    if args.cuda:
        features = features.cuda()
        adj1 = adj1.cuda()
        adj2 = adj2.cuda()
        model.cuda()
        VGAE1.cuda()
        VGAE2.cuda()

    if args.initcluster == "kmeans":
        print("Initializing cluster centers with kmeans, n_clusters known")
        kmeans = KMeans(args.n_cluster, n_init=20, algorithm='elkan')
        y_pred = kmeans.fit_predict(emb.detach().cpu().numpy())

    elif args.initcluster == "louvain":
        print("Initializing cluster centers with louvain,resolution=", args.res)
        adata = sc.AnnData(emb.detach().cpu().numpy())
        sc.pp.neighbors(adata, n_neighbors=args.n_cluster)
        sc.tl.louvain(adata, resolution=args.res)
        y_pred = adata.obs['louvain'].astype(int).to_numpy()
        n = len(np.unique(y_pred))

    emb = pd.DataFrame(emb.detach().cpu().numpy(), index=np.arange(0, emb.shape[0]))
    Group = pd.Series(y_pred, index=np.arange(0, emb.shape[0]), name="Group")
    Mergefeature = pd.concat([emb, Group], axis=1)
    cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())

    y_pred_last = y_pred
    with torch.no_grad():
        model.cluster_layer.copy_(torch.tensor(cluster_centers))

    best_adj_acc = 0
    adj_acc = 0.1

    model.train()
    VGAE1.train()
    VGAE2.train()
    for epoch in tqdm(range(args.max_epochs)):
        pred1, mu1, log_sigma1 = VGAE1(features, adj1)
        pred2, mu2, log_sigma2 = VGAE2(features, adj2)

        fVGAE_loss = VGAE_Loss(preds=pred1, labels=adj1_label, mu=mu1, logvar=log_sigma1,
                               n_nodes=features.shape[0], norm=norm1, pos_weight=pos_weight1)
        optimizer_vgae1.zero_grad()
        fVGAE_loss.backward()
        optimizer_vgae1.step()

        sVGAE_loss = VGAE_Loss(preds=pred2, labels=adj2_label, mu=mu2, logvar=log_sigma2,
                               n_nodes=features.shape[0], norm=norm2, pos_weight=pos_weight2)
        optimizer_vgae2.zero_grad()
        sVGAE_loss.backward()
        optimizer_vgae2.step()

        if adj_acc > best_adj_acc:
            pred1, mu1, log_sigma1 = VGAE1(features, adj1)
            pred1 = pred1.detach()
            mu1_best = mu1.detach()
            sigma1 = torch.exp(log_sigma1.detach())

            pred2, mu2, log_sigma2 = VGAE2(features, adj2)
            pred2 = pred2.detach()
            mu2_best = mu2.detach()
            sigma2 = torch.exp(log_sigma2.detach())

            best_adj_acc = adj_acc

        fContrastive_loss = Implicit_Contrastive_Loss(Z=emb1, mu=mu1_best, sigma2=sigma1 ** 2,
                                                      tau=args.tau, num_samples=args.num_samples,
                                                      device=args.device)
        sContrastive_loss = Implicit_Contrastive_Loss(Z=emb2, mu=mu2_best, sigma2=sigma2 ** 2,
                                                      tau=args.tau, num_samples=args.num_samples,
                                                      device=args.device)
        Contrastive_loss = fContrastive_loss + sContrastive_loss

        fadj_acc = Reconstruct_Ratio(pred1.cpu(), adj1_ori)
        sadj_acc = Reconstruct_Ratio(pred2.cpu(), adj2_ori)
        adj_acc = fadj_acc + sadj_acc

        if epoch % args.update_interval == 0:
            tem_q = model(features, adj1, adj2)[-1]
            tem_q = tem_q.detach()
            p = target_distribution(tem_q)

            y_pred = torch.argmax(tem_q, dim=1).cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            # y = labels

            # nmi = nmi_score(y, y_pred)
            # ari = ari_score(y, y_pred)

            if epoch > 0 and delta_label < args.tol:
                print('delta_label ', delta_label, '< tol ', args.tol)
                print("Reach tolerance threshold. Stopping training.")
                break

        _, _, x, q = model(features, adj1, adj2)
        loss_model = kl_loss(q.log(), p) + 0.1 * Contrastive_loss

        optimizer_model.zero_grad()
        loss_model.backward()
        optimizer_model.step()

        emb1, emb2, _ = model.mgcn(features, adj1, adj2)

    model.eval()
    _, _, x, _ = model(features, adj1, adj2)
    adata.obsm['STMIGCL'] = x.detach().numpy()
    adata.obs['pred'] = y_pred
    adata.obs["pred"] = adata.obs["pred"].astype('category')
    return adata
