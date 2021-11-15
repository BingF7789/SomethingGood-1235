import argparse
import pickle as pk
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import sklearn as sk


def arg_parameter():
    parser = argparse.ArgumentParser()
    # Training arguments
    parser.add_argument('--device', type=str, default='cpu', help='')
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--adj_data', type=str, default='data/sensor_graph/adj_mx.pkl', help='adj data path')
    parser.add_argument('--unequal', type=int, default=0, help='whether to use unequal data splits')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--debug', type=int, default=0, help='debug mode')
    parser.add_argument('--reg', type=int, default=1, help='enable regulizer or not for local train')
    parser.add_argument('--com_round', type=int, default=20, help='Number of communication round to train.')
    parser.add_argument('--epoch', type=int, default=10, help='epoch for each communication round.')
    parser.add_argument('--logDir', default='./log/,default.txt', help='Path for log info')
    parser.add_argument('--num_thread', type=int, default=10, help='number of threading to use for client training.')

    # Federated arguments
    parser.add_argument('--clients', type=int, default=100, help="number of users: K")
    parser.add_argument('--shards', type=int, default=2, help="each client roughly have 2 data classes")
    parser.add_argument('--serveralpha', type=float, default=0.1, help='server prop alpha')
    parser.add_argument('--serverbeta', type=float, default=0.3, help='personalized agg rate alpha')
    parser.add_argument('--deep', type=int, default=0, help='0: 1 layer only, 1: 2 layers, 3:full-layers')
    parser.add_argument('--agg', type=str, default='none', help='averaging strategy')
    parser.add_argument('--dp', type=float, default=0.001, help='differential privacy')
    parser.add_argument('--epsilon', type=float, default=1, help='stepsize')
    parser.add_argument('--ord', type=int, default=2, help='similarity metric')
    # Graph Learning
    parser.add_argument('--subgraph_size', type=int, default=30, help='k')
    parser.add_argument('--tanhalpha', type=float, default=3, help='adj alpha')
    parser.add_argument('--gc_epoch', type=int, default=10, help='')

    # CNN tasks related
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imgs")
    parser.add_argument('--hidden', type=str, default="10,20,320,50", help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--bn', type=int, default=0, help="enable batch norm of CNN model")

    # RNN tasks related
    parser.add_argument('--clip', type=int, default=5, help='clip')
    parser.add_argument('--step_size1', type=int, default=2500, help='step_size')
    parser.add_argument('--step_size2', type=int, default=100, help='step_size')
    parser.add_argument('--rnn_hidden', type=int, default=64, help='Number of rnn hidden size.')

    # MLP tasks related

    # Others
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets -- 32 for mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True', help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--print_every', type=int, default=100, help='')
    parser.add_argument('--save', type=str, default='./save/', help='save path')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--valid_freq', type=int, default=1, help='validation at every n communication round')
    # model related valid_freq
    # parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
    # parser.add_argument('--buildA_true', type=str_to_bool, default=True,
    #                     help='whether to construct adaptive adjacency matrix')
    # parser.add_argument('--load_static_feature', type=str_to_bool, default=False, help='whether to load static feature')
    parser.add_argument('--cl', type=int, default=1, help='whether to do curriculum learning')
    parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
    parser.add_argument('--dilation_exponential', type=int, default=1, help='dilation exponential')
    parser.add_argument('--conv_channels', type=int, default=32, help='convolution channels')
    parser.add_argument('--residual_channels', type=int, default=32, help='residual channels')
    parser.add_argument('--skip_channels', type=int, default=64, help='skip channels')
    parser.add_argument('--end_channels', type=int, default=128, help='end channels')
    parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
    parser.add_argument('--seq_in_len', type=int, default=12, help='input sequence length')
    parser.add_argument('--seq_out_len', type=int, default=12, help='output sequence length')
    parser.add_argument('--layers', type=int, default=3, help='number of layers')
    parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
    parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
    parser.add_argument('--runs', type=int, default=10, help='number of runs')

    # GAE model related
    parser.add_argument('--alpha', type=int, default=100, help='')
    parser.add_argument('--gae_mode', type=int, default=1, help='0 to reconstract A, 1 for X')

    args = parser.parse_args()

    hidden = []
    for h in args.hidden.split(","):
        hidden.append(int(h))
    args.hidden = hidden
    logs = args.logDir.split(",")
    args.logDir = logs[0] + args.dataset + "-" + args.agg + "-" + \
                  str(args.com_round) + "-" + str(args.epoch) + "-" + logs[1]
    return args


def mtx_similar3(arr1:np.ndarray, arr2:np.ndarray) ->float:
    '''
    reshape the matrices into vectors and compute the Euclidean distance between them.
    difference = np.linalg.norm(dists - dists_one, ord='fro')
    '''

    farr1 = arr1.ravel()
    farr2 = arr2.ravel()
    len1 = len(farr1)
    len2 = len(farr2)
    if len1 > len2:
        farr1 = farr1[:len2]
    else:
        farr2 = farr2[:len1]

    numer = np.sum(farr1 * farr2)
    denom = np.sqrt(np.sum(farr1 ** 2) * np.sum(farr2 ** 2))
    similar = numer / denom
    return round(((similar + 1) / 2) * 100, 2)


def mtx_similar(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1


def sparse_adj(adj):
    # mask = torch.zeros(100, 100)
    mask = torch.zeros(207, 207)
    mask.fill_(float('0'))
    s1, t1 = adj.topk(60, 1)
    mask.scatter_(1, t1, s1.fill_(1))
    adj = adj * mask

    return adj

# args = arg_parameter()
# adj_file = args.logDir.replace("log", "learned_adj")[:-4] + ".pth"

x_label = "METR-LA Similarity: "
adj_file = "METR-LA-graph_v2-20-10-1005-001.pth"

# x_label = "PEMS-BAY Similarity: "
# adj_file = "PEMS-BAY-graph_v2-20-10-1005-002.pth"

# x_label = "MNIST Similarity: "
# adj_file = "./learned_adj/mnist-graph_v3-20-20-1217-027.pth"

# x_label = "CIFAR-10 Similarity: "
# adj_file = "./learned_adj/cifar10-graph_v3-20-10-1203-002.pth"

# data = torch.load(adj_file, map_location=torch.device('cpu'))
with open(adj_file, "rb") as fw:
    data = pk.load(fw)

pre_adj = data["pre_adj"][0:10, 0:20].cpu().numpy()
learned_adj = data["adj"][0:10, 0:20].cpu().numpy()

# sim = mtx_similar3(sparse_adj(data["pre_adj"].cpu()).numpy(), data["adj"].numpy())
sim = mtx_similar3(data["pre_adj"].cpu().numpy(), data["adj"].numpy())


# precision, recall, f1 = mtx_similar3(sparse_adj(data["pre_adj"]).numpy().flatten(), data["adj"].numpy().flatten())


# vmin = min((pre_adj.min()), learned_adj.min())
# vmax = max(pre_adj.max(), learned_adj.max())

vmin1 = pre_adj.min()
vmin2 = learned_adj.min()

vmax1 = pre_adj.max()
vmax2 = learned_adj.max()

fig, axs = plt.subplots(nrows=2, gridspec_kw=dict(height_ratios=[5, 5]))

sns.heatmap(pre_adj * 1.5, annot=False, ax=axs[0], vmin=vmin1, vmax=vmax1, cmap="YlGnBu",
            cbar=False, xticklabels=False, yticklabels=False)

sns.heatmap(learned_adj*0.5, annot=False, ax=axs[1], vmin=vmin2, vmax=vmax2, cmap="YlGnBu",
            cbar=False, xticklabels=False, yticklabels=False)

axs[0].set(ylabel="Pre-Defined")
axs[0].set_title('Similarity: ' + str(sim) + "%")
axs[1].set(ylabel="Learned")
axs[1].set(xlabel=x_label + str(sim) + "%")
# axs[1].set(xlabel="MNIST Similarity: 77.23%")
# axs[1].set(xlabel="Precision:" + str(precision) + " Recall:" + str(recall) + " F1: " + str(f1))

plt.show()
