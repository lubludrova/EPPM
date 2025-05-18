import torch

from src.model.graph_layers.GraphConv import GraphConv
from src.model.graph_layers.Pooling import SumPool, MaxPool
from src.model.layers.GGRNN import GGRNN


class CNNLSTM(torch.nn.Module):

    def __init__(self, hyperparams, N, F, unique_activities, log_name,
                 max_len, vectorizer, args, attribute_count, is_search):
        super().__init__()
        self.N = N
        self.F = F
        self.unique_activities = unique_activities
        self.log_name = log_name
        #self.max_len = max_len
        self.vectorizer = vectorizer
        self.is_search = is_search
        self.args = args
        self.attribute_count = attribute_count
        self.attribute_count = len(list(attribute_count.keys()))
        self.attributes = attribute_count
        self.max_len = max_len

        self.rnn_1_hidden_size = hyperparams["rnn_1"]
        #self.rnn_2_hidden_size = hyperparams["rnn_2"]
        self.gnn_hidden_size = hyperparams["gnn_1"]
        self.dropout = hyperparams["dropout"]

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.embedding_size = hyperparams["embedding"]

        print("Attribute count: ", self.attribute_count)

        n_embeddings = 5 + self.attribute_count
        self.embedded_F = self.F - n_embeddings + n_embeddings * self.embedding_size
        print("N embedding: ", n_embeddings)
        print("Self emb f: ", self.embedded_F)
        print("Self F: ", self.F)

        hidden_size = 128
        self.dense_act = torch.nn.Linear(hidden_size, len(self.unique_activities) + 1).to(self.device)
        #self.graph_conv = GraphConv(hidden_size, self.N, hidden_size).to(self.device)
        self.lstm = torch.nn.GRU(hidden_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2).to(self.device)
        self.conv1 = torch.nn.Conv2d(self.embedded_F, hidden_size, kernel_size=(1, 5)).to(self.device)
        self.conv2 = torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 3)).to(self.device)
        self.dense = torch.nn.Linear(hidden_size, hidden_size).to(self.device)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, concatenation, A_in, multiple_outputs):
        concatenation = concatenation.permute(0, 3, 1, 2)

        concatenation = torch.nn.ZeroPad2d((0, 0, self.max_len - concatenation.shape[2], 0))(concatenation)

        c = self.conv1(concatenation)

        c = torch.nn.ReLU()(c)

        c = self.conv2(c)

        c = torch.nn.ReLU()(c)

        #binded = []
        #for unb in torch.unbind(c, dim=2):
        #    binded.append(self.graph_conv([unb.permute(0, 2, 1), A_in]).permute(0, 2, 1)[:, :, None, :])
        #binded = torch.cat(binded, dim=2)

        bind = torch.nn.MaxPool2d(kernel_size=(c.shape[2], 1))(c)

        bind = bind[:, :, 0, :].permute(0, 2, 1)

        c = self.lstm(bind)
        outputs = c[0][:, -1, :]

        outputs = self.dense(outputs)

        outputs = self.dropout(outputs)

        outputs = torch.nn.ReLU()(outputs)

        outputs = [self.dense_act(outputs)]


        if multiple_outputs:
            return outputs
        else:
            return outputs[0]


