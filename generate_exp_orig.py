import sys
import pickle
import torch
from dgl.dataloading import GraphDataLoader
from preprocessing.dgl_dataset import TextDataset
import time
import h5py
from dgl.nn.pytorch.explain import HeteroGNNExplainer
import os
log_name = 'Helpdesk.xes.gz'
print(log_name, '-----')


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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    start_time = time.time()

    base_dir = '/Users/ruslanageev/PycharmProjects/Prophet'
    X_test, y_test = load_graphs_from_hdf5(
        os.path.join(base_dir, 'heterographs_tracenode', f"{log_name}_test.db"))

    with open("/Users/ruslanageev/PycharmProjects/Prophet/preprocessing/w2v/" + log_name + "/relations.pkl", 'rb') as f:
        relation = pickle.load(f)

    number_of_examples = 10
    model = torch.load(f'{base_dir}/models/model_{log_name}.h5')

    graphs = X_test[:number_of_examples]
    labels = y_test[:number_of_examples]
    print(X_test[0], y_test[0])


    df_test = TextDataset(X_test[:number_of_examples], y_test[:number_of_examples])
    test_loader = GraphDataLoader(df_test,
                                      batch_size=1,
                                      drop_last=False,
                                      shuffle=False)

    explainer = HeteroGNNExplainer(model.to(device), num_hops=1, log=False)

    dict_feature = {}
    dict_edge = {}
    for n in next(iter(test_loader))[0].ntypes:
        dict_feature[n] = []
    for e in relation:
        dict_edge[e] = []
    i = 0
    for g, l in test_loader:
        print(i)
        g = g.to(device)
        l = l.to(device)
        feat = {}
        for n in g.ntypes:
            feat[n] = g.ndata[n][n]

        feat_mask, edge_mask = explainer.explain_graph(graph=g, feat=feat)
        for f in g.ntypes:
            val = feat_mask[f]
            dict_feature[f].append(val)

        for e in relation:
            dict_edge[e].append(edge_mask[e])
        i = i +1


    for f in next(iter(test_loader))[0].ntypes:
        with open(f'utility/{log_name}_feature_{f}.pickle', 'wb') as fil:
            pickle.dump(dict_feature[f], fil)

    for e in relation:
        with open(f'utility/{log_name}_edge_{e}.pickle', 'wb') as fil2:
            pickle.dump(dict_edge[e], fil2)