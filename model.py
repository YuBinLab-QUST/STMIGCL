import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from layer import GraphConvolution



class GCN(nn.Module):
    def __init__(self, nfeat, out):
        super(GCN, self).__init__()
        self.gc = GraphConvolution(nfeat, out)

    def forward(self, x, adj):
        x = self.gc(x, adj)
        return x


class Attention(nn.Module):
    def __init__(self, in_size):
        super(Attention, self).__init__()
        self.project = nn.Linear(in_size, 1, bias=False)

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)


class MGCN(nn.Module):
    def __init__(self, nfeat, nemb):
        super(MGCN, self).__init__()
        self.GCNA1 = GCN(nfeat, nemb)
        self.GCNA2 = GCN(nfeat, nemb)
        self.attention = Attention(nemb)

    def forward(self, x, adj1, adj2):
        emb1 = self.GCNA1(x, adj1)
        emb2 = self.GCNA2(x, adj2)
        emb = torch.stack([emb1, emb2], dim=1)
        emb = self.attention(emb)
        return emb1, emb2, emb


class STMGCN(nn.Module):
    def __init__(self, nfeat, nemb, nclass):
        super(STMGCN, self).__init__()
        self.mgcn = MGCN(nfeat, nemb)
        self.cluster_layer = Parameter(torch.Tensor(nclass, nemb))
        torch.nn.init.xavier_normal_(self.cluster_layer)

    def forward(self, x, adj1, adj2):
        emb1, emb2, x = self.mgcn(x, adj1, adj2)
        self.alpha = 0.2
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.cluster_layer) ** 2, dim=2) / self.alpha))
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = q ** (self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return emb1, emb2, x, q
