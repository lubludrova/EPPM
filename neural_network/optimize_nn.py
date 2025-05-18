import torch
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if torch.backends.mps.is_available():        # Mac + Metal
#     device = torch.device("mps")
if torch.cuda.is_available():              # NVIDIA GPU
    device = torch.device("cuda")
else:                                        # CPU‑резерв
    device = torch.device("cpu")
import pickle
from preprocessing.dgl_dataset import TextDataset
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from dgl.dataloading import GraphDataLoader
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from neural_network.hetero_gnn import HeteroClassifier
from time import perf_counter
import time
import sys
import h5py

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

space = {'hidden_dim': hp.choice('hidden_dim', [64]),
         'dropout': hp.choice("dropout", [0.15]),
         'n_heads': hp.choice('n_heads', [4]),
         'batch_size': hp.choice('batch_size', [8]),
         'learning_rate': hp.choice("learning_rate", [0.001]),
         'n_layers': hp.choice('n_layers', [1]),
         }


criterion = torch.nn.CrossEntropyLoss()

def train_fn(model, X_train_batch, y_train_batch, optimizer, criterion):
        model.train()
        X_train_batch = X_train_batch.to(device)
        y_train_batch = y_train_batch.to(device)

        optimizer.zero_grad()
        feature = {}
        for n in X_train_batch.ntypes:
            feature[n] = X_train_batch.ndata[n][n]
        eweight = None
        y_train_pred = model(X_train_batch, feature, eweight)
        loss = criterion(y_train_pred, y_train_batch)
        loss.backward()
        optimizer.step()
        return loss.item()

def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)
            feature = {}
            for n in X.ntypes:
                feature[n] = X.ndata[n][n]
            eweight = None
            y_pred = model(X, feature, eweight)
            loss = criterion(y_pred, y)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)


def train_gnn(model, train_data_loader, valid_data_loader, optimizer, EPOCHS):
    best_valid_loss = float("inf")
    early_stop_counter = 0
    patience = 20
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(EPOCHS):
        train_loss = 0
        for x_batch, y_batch in train_data_loader:
            batch_loss = train_fn(model.to(device), x_batch, y_batch, optimizer, criterion)
            train_loss += batch_loss

        avg_train_loss = train_loss / len(train_data_loader)
        valid_loss = evaluate_fn(model.to(device),valid_data_loader,criterion,device)
        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = model
            early_stop_counter = 0  # Reset early stopping counter
        else:
            early_stop_counter += 1

        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {valid_loss:.4f}", end=' ')
        if early_stop_counter >= patience:
            print("Validation loss hasn't improved for", patience, "epochs. Early stopping...")
            break
    return best_model, best_valid_loss


def get_model(params, train_loader):
    g = next(iter(train_loader))[0]
    f_map = {ntype: (g.nodes[ntype[0]].data[ntype[0]].shape[1], g.nodes[ntype[2]].data[ntype[2]].shape[1]) for ntype in
             g.canonical_etypes}  # for SageConv
    model = HeteroClassifier(f_map, params, len(y_train[0]), g.canonical_etypes).to(device)
    return model


def fit_and_score(params):
    print(params)
    start_time = perf_counter()
    train_loader = GraphDataLoader(df_train,
                                   batch_size=2 ** params['batch_size'],
                                   drop_last=False,
                                   shuffle=True,
                                   )

    val_loader = GraphDataLoader(df_val,
                                 batch_size=2 ** params['batch_size'],
                                 drop_last=False,
                                 shuffle=True,
                                 )

    model = get_model(params, train_loader)
    print(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    model, score = train_gnn(model, train_loader, val_loader, optimizer, 20 )

    global best_score, best_model, best_time, best_numparameters
    end_time = perf_counter()

    if best_score > score:
        best_score = score
        best_model = model
        best_numparameters = sum(param.numel() for param in model.parameters())
        best_time = end_time - start_time
        torch.save(best_model, '/Users/ruslanageev/PycharmProjects/Prophet/models/model_' + log_name + '.h5')

    return {'loss': score, 'status': STATUS_OK, 'n_params': sum(param.numel() for param in model.parameters()),
            'time': end_time - start_time}


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

if __name__ == "__main__":
    # log_name = "BPI_Challenge_2012_O.xes.gz"
    # log_name = "env_permit.xes.gz"
    # log_name = "BPI_ Challenge_2012_W.xes.gz"
    # log_name = "BPI_Challenge_2013_closed_problems.xes.gz"
    log_name = "SEPSIS.xes.gz"

    current_time = time.strftime("%d.%m.%y-%H.%M", time.localtime())

    outfile = open(f'outfile_{log_name}.txt', 'w')
    # device = torch.device("c  uda:0" if torch.cuda.is_available() else "cpu")
    # if torch.backends.mps.is_available():  # Mac + Metal
    #     device = torch.device("mps")
    if torch.cuda.is_available():  # NVIDIA GPU
        device = torch.device("cuda")
    else:  # CPU‑резерв
        device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    #conn = sqlite3.connect('heterographs_tracenode/' +log_name+'_train.db')
    #c = conn.cursor()
    outfile.write("Starting time: %s\n" % current_time)

    # Load graphs

    X_train, y_train = load_graphs_from_hdf5('/Users/ruslanageev/PycharmProjects/Prophet/heterographs_tracenode/' + log_name + '_train.db')
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)

    df_train = TextDataset(X_train, y_train)
    df_val = TextDataset(X_val, y_val)

    print('Starting model selection...')
    best_score = np.inf
    best_model = None
    best_time = 0
    best_numparameters = 0

    trials = Trials()
    best = fmin(fit_and_score, space, algo=tpe.suggest, max_evals=1, trials=trials,
                rstate=np.random.RandomState(seed))
    best_params = hyperopt.space_eval(space, best)

    outfile.write("\nHyperopt trials")
    outfile.write("\ntid,loss,learning_rate,batch_size,n_heads,agg,time,n_params,perf_time,hidden_dim,n_layers")
    for trial in trials.trials:
        outfile.write("\n%d,%f,%f,%d,%s,%s,%d,%f,%d,%d" % (trial['tid'],
                                                           trial['result']['loss'],
                                                           trial['misc']['vals']['learning_rate'][0],
                                                           trial['misc']['vals']['batch_size'][0] + 6,
                                                           trial['misc']['vals']['n_heads'][0] + 1,
                                                           (trial['refresh_time'] - trial['book_time']).total_seconds(),
                                                           trial['result']['n_params'],
                                                           trial['result']['time'],
                                                           trial['misc']['vals']['hidden_dim'][0],
                                                           trial['misc']['vals']['n_layers'][0] + 1

                                                           ))
    outfile.write("\n\nBest parameters:")
    print(best_params, file=outfile)
    outfile.write("\nModel parameters: %d" % best_numparameters)
    outfile.write('\nBest Time taken: %f' % best_time)
