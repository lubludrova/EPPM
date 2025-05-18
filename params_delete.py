import sys
import pickle

import numpy as np
import pandas as pd
import torch
from dgl.dataloading import GraphDataLoader
from torch import nn

from preprocessing.dgl_dataset import TextDataset
import time
import h5py
from dgl.nn.pytorch.explain import HeteroGNNExplainer, HeteroPGExplainer, HeteroSubgraphX
from metrics_xai import compute_xai_metrics, visualize_explanation
import torch.nn.functional as F
import random
import torch
import networkx as nx
import matplotlib.pyplot as plt
from dgl import to_networkx
import os
import ast

from utility.TAPG import TypeAwarePGExplainer

log_name = 'Helpdesk.xes.gz' #'Helpdesk.xes.gz'
print(log_name, '-----')
device = torch.device("cpu")

def load_graphs_from_hdf5(filename):
    graphs = []
    label = []
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            pickled_graph = f[key][()]
            graph = pickle.loads(pickled_graph)
            graphs.append(graph['graph'])
            label.append(graph['label'])
    return graphs, label





if __name__ == '__main__':
    need_compute = True
    explain = 'pg' # pg, gnn, attention, tapg, subx


    base_dir = '/Users/ruslanageev/PycharmProjects/Prophet'
    dump_dir = '/Users/ruslanageev/PycharmProjects/Prophet/utility/files'
    number_of_examples = 13

    start_time = time.time()

    model = torch.load(f'{base_dir}/models/model_{log_name}.h5')
    n_layers = len(model.rgcn.convs)
    # возьмём первый слой и первый тип отношения
    first_conv = next(iter(model.rgcn.convs[0].mods.values()))
    n_heads = first_conv._num_heads
    hidden_dim = first_conv._out_feats
    # dropout = first_conv._drop
    print(n_layers, n_heads, hidden_dim)
    # model.eval()
