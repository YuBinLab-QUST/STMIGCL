import numpy as np
import torch
import torch.nn.functional as F


def target_distribution(q):
    p = q ** 2 / torch.sum(q, dim=0)
    p = p / torch.sum(p, dim=1, keepdim=True)
    return p


def kl_loss(q, p):
    return F.kl_div(q, p, reduction="batchmean")


def VGAE_Loss(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD


def Implicit_Contrastive_Loss(Z, mu, sigma2, tau, num_samples, device):
    sampled_indices = np.random.choice(Z.size(0), num_samples, replace=False)
    assert len(set(sampled_indices)) == len(sampled_indices)
    Z = Z[sampled_indices]
    mu = mu[sampled_indices]
    sigma2 = sigma2[sampled_indices]

    N = Z.size(0)
    C = Z.size(0)
    L = Z.size(1)

    y_t = torch.arange(start=0, end=N).to(device)

    NxW_ij = Z.expand(N, C, L)
    sigma2_expand = sigma2.expand(N, C, L)

    NxW_kj = torch.gather(NxW_ij,
                          1,
                          y_t.view(N, 1, 1).expand(N, C, L))

    quadra = (sigma2_expand * (NxW_ij - NxW_kj) ** 2).sum(dim=2)
    dot_sim = Z @ (mu.T) / tau

    y_aux = dot_sim + 0.5 / tau ** 2 * quadra
    loss = F.cross_entropy(y_aux, y_t)
    return loss
