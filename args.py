import os
import torch
import argparse
import numpy as np


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--dataset', default='DLPFC', type=str)
    parser.add_argument('--slice', default="151507", type=str)
    parser.add_argument('--lr1', default=0.003, type=float)
    parser.add_argument('--lr2', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--res', default=1, type=float)

    parser.add_argument('--initcluster', default="kmeans", type=str)
    parser.add_argument('--n_cluster', default=7, type=int)
    parser.add_argument('--max_epochs', default=2000, type=int)
    parser.add_argument('--update_interval', default=3, type=int)
    parser.add_argument('--tol', default=0.0001, type=float)

    parser.add_argument('--tau', default=1, type=float)
    parser.add_argument('--num_samples', default=100, type=int)

    parser.add_argument('--npca', default=30, type=int)
    parser.add_argument('--nemb', default=15, type=int)
    parser.add_argument('--k', default=20, type=int)
    parser.add_argument('--radius', default=700, type=int)

    args = parser.parse_args(args=[])

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.cuda = torch.cuda.is_available()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)

    return args
