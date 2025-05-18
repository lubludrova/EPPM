import pickle

import pandas as pd
import torch
from dgl.dataloading import GraphDataLoader
from preprocessing.dgl_dataset import TextDataset
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
import sys
import h5py


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_metrics(y_true, y_pred):
    return {
        'accuracy':  accuracy_score (y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall':    recall_score   (y_true, y_pred, average='macro'),
        'f1':        f1_score       (y_true, y_pred, average='macro'),
    }

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
    log_name = 'Helpdesk.xes.gz'
    print(log_name, '-----')

    # Load graphs
    X_test, y_test = load_graphs_from_hdf5('/Users/ruslanageev/PycharmProjects/Prophet/heterographs_tracenode/' + log_name + '_test.db')

    model = torch.load(f'/Users/ruslanageev/PycharmProjects/Prophet/models/model_{log_name}.h5')
    df_test = TextDataset(X_test, y_test)

    test_loader = GraphDataLoader(df_test,
                                  batch_size=256,
                                  drop_last=False,
                                  shuffle=False)

    list_pred = []
    list_truth = []

    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            list_edge = X.edges(etype=('concept:name', 'follow', 'concept:name'))
            feature = {}
            for n in X.ntypes:
                feature[n] = X.ndata[n][n]
            # model.rgcn(X, feature)
            pred = model(X, feature).argmax(dim=1)
            list_pred.extend(pred.cpu().numpy())
            list_truth.extend(y.argmax(dim=1).cpu().numpy())
            # print(list_pred, list_truth)

    lengths = [g.number_of_nodes('concept:name') for g in X_test]

    df = pd.DataFrame({
        'truth': list_truth,
        'pred': list_pred,
        'length': lengths
    })
    print(df)
    metrics_per_group = []
    for grp, sub in df.groupby('length'):
        m = compute_metrics(sub['truth'], sub['pred'])
        m['group'] = grp
        m['n_samples'] = len(sub)
        print(m)
        metrics_per_group.append(m)

    metrics_df = pd.DataFrame(metrics_per_group)[
        ['group', 'n_samples', 'accuracy', 'precision', 'recall', 'f1']
    ]
    print(metrics_df.to_string(index=False))

    print(accuracy_score(df['truth'], df['pred']))

    # precision, recall, fscore, _ = precision_recall_fscore_support(list_truth, list_pred, average='macro',
    #                                                                pos_label=None)
    # print("fscore-->{:.3f}".format(fscore))
