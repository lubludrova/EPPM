import torch

from src.model.graph_layers.GraphConv import GraphConv


class LSTMGNNCell(torch.nn.Module):
    def __init__(self, n_hidden, N, X_shape, dropout):
        super().__init__()
        self.n_hidden = n_hidden
        self.N = N
        self.dropout = dropout

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.conv_ui = GraphConv(n_hidden, N, X_shape).to(self.device)
        self.conv_wi = GraphConv(n_hidden, N, n_hidden).to(self.device)
        self.conv_vi = GraphConv(n_hidden, N, n_hidden).to(self.device)
        self.conv_uf = GraphConv(n_hidden, N, X_shape).to(self.device)
        self.conv_wf = GraphConv(n_hidden, N, n_hidden).to(self.device)
        self.conv_vf = GraphConv(n_hidden, N, n_hidden).to(self.device)
        self.conv_ug = GraphConv(n_hidden, N, X_shape).to(self.device)
        self.conv_wg = GraphConv(n_hidden, N, n_hidden).to(self.device)
        self.conv_uo = GraphConv(n_hidden, N, X_shape).to(self.device)
        self.conv_wo = GraphConv(n_hidden, N, n_hidden).to(self.device)
        self.conv_vo = GraphConv(n_hidden, N, n_hidden).to(self.device)

        self.bias_i = torch.nn.Parameter(data=torch.Tensor(N, n_hidden), requires_grad=True)
        self.bias_f = torch.nn.Parameter(data=torch.Tensor(N, n_hidden), requires_grad=True)
        self.bias_g = torch.nn.Parameter(data=torch.Tensor(N, n_hidden), requires_grad=True)
        self.bias_o = torch.nn.Parameter(data=torch.Tensor(N, n_hidden), requires_grad=True)

        # self.batch_norm_z = torch.nn.BatchNorm1d(N)
        # self.batch_norm_r = torch.nn.BatchNorm1d(N)

        #self.batch_norm_i = torch.nn.LayerNorm(n_hidden)
        #self.batch_norm_f = torch.nn.LayerNorm(n_hidden)
        #self.batch_norm_g = torch.nn.LayerNorm(n_hidden)
        #self.batch_norm_o = torch.nn.LayerNorm(n_hidden)

        if self.dropout > 0:
            self.dropout_z = torch.nn.Dropout(self.dropout)
            self.dropout_r = torch.nn.Dropout(self.dropout)

        torch.nn.init.ones_(self.bias_i)
        torch.nn.init.ones_(self.bias_o)
        torch.nn.init.ones_(self.bias_f)
        torch.nn.init.ones_(self.bias_g)

    def forward(self, x, hidden):
        if hidden is not None:
            h_t_1 = hidden[0]
            c_t_1 = hidden[1]
        else:
            h_t_1 = torch.zeros((self.N, self.n_hidden)).to(self.device)
            c_t_1 = torch.zeros((self.N, self.n_hidden)).to(self.device)
        X = x[0]
        A = x[1]

        i = self.conv_ui([X, A]) + self.conv_wi([h_t_1, A]) + self.conv_vi([c_t_1, A]) + self.bias_i
        i = torch.sigmoid(i)

        f = self.conv_uf([X, A]) + self.conv_wf([h_t_1, A]) + self.conv_vf([c_t_1, A]) + self.bias_f
        f = torch.sigmoid(f)

        o = self.conv_uo([X, A]) + self.conv_wo([h_t_1, A]) + self.conv_vo([c_t_1, A]) + self.bias_o
        o = torch.sigmoid(o)

        c_vir = torch.tanh(self.conv_ug([X, A]) + self.conv_wg([h_t_1, A]) + self.bias_g)
        c_vir = torch.tanh(c_vir)

        c = torch.sigmoid(f * c_t_1 + i * c_vir)

        h = torch.tanh(c) * o

        return h, c
