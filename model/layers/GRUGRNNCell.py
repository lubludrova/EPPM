import torch

from src.model.graph_layers.GraphAttention import GraphAttention
from src.model.graph_layers.GraphConv import GraphConv
from src.model.graph_layers.GCNII import GCNII


class GRUGNNCell(torch.nn.Module):
    def __init__(self, n_hidden, N, X_shape, dropout, n_layer=1, is_last=True):
        super().__init__()
        self.n_hidden = n_hidden
        self.N = N
        self.dropout = dropout
        self.is_last = is_last

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.conv_z_1 = GraphConv(n_hidden, N, X_shape).to(self.device)
        self.conv_z_2 = GraphConv(n_hidden, N, n_hidden).to(self.device)
        self.conv_r_1 = GraphConv(n_hidden, N, X_shape).to(self.device)
        self.conv_r_2 = GraphConv(n_hidden, N, n_hidden).to(self.device)
        self.conv_h_1 = GraphConv(n_hidden, N, X_shape).to(self.device)
        self.conv_h_2 = GraphConv(n_hidden, N, n_hidden).to(self.device)

        """
        self.conv_z_1 = GCNII(n_hidden, N, X_shape, n_layer=n_layer, is_last=is_last).to(self.device)
        self.conv_z_2 = GCNII(n_hidden, N, n_hidden, n_layer=n_layer, is_last=is_last).to(self.device)
        self.conv_r_1 = GCNII(n_hidden, N, X_shape, n_layer=n_layer, is_last=is_last).to(self.device)
        self.conv_r_2 = GCNII(n_hidden, N, n_hidden, n_layer=n_layer, is_last=is_last).to(self.device)
        self.conv_h_1 = GCNII(n_hidden, N, X_shape, n_layer=n_layer, is_last=is_last).to(self.device)
        self.conv_h_2 = GCNII(n_hidden, N, n_hidden, n_layer=n_layer, is_last=is_last).to(self.device)
        """

        """
        self.conv_z_1 = GraphAttention(n_hidden, N, X_shape).to(self.device)
        self.conv_z_2 = GraphAttention(n_hidden, N, n_hidden).to(self.device)
        self.conv_r_1 = GraphAttention(n_hidden, N, X_shape).to(self.device)
        self.conv_r_2 = GraphAttention(n_hidden, N, n_hidden).to(self.device)
        self.conv_h_1 = GraphAttention(n_hidden, N, X_shape).to(self.device)
        self.conv_h_2 = GraphAttention(n_hidden, N, n_hidden).to(self.device)
        """

        self.bias_z = torch.nn.Parameter(data=torch.Tensor(N, n_hidden), requires_grad=True)
        self.bias_r = torch.nn.Parameter(data=torch.Tensor(N, n_hidden), requires_grad=True)
        self.bias_h = torch.nn.Parameter(data=torch.Tensor(N, n_hidden), requires_grad=True)

        # self.batch_norm_z = torch.nn.BatchNorm1d(N)
        # self.batch_norm_r = torch.nn.BatchNorm1d(N)

        #self.layer_normalization_z = torch.nn.LayerNorm(n_hidden)
        #self.layer_normalization_r = torch.nn.LayerNorm(n_hidden)
        #self.layer_normalization_h = torch.nn.LayerNorm(n_hidden)

        if self.dropout > 0:
            self.dropout_layer = torch.nn.Dropout(self.dropout)

        torch.nn.init.ones_(self.bias_z)
        torch.nn.init.ones_(self.bias_r)
        torch.nn.init.ones_(self.bias_h)

    def forward(self, x, hidden):
        X = x[0].to('cpu')
        A = x[1].to('cpu')
        if hidden is not None:
            h_t_1 = hidden
        else:
            h_t_1 = torch.zeros((X.shape[0], self.N, self.n_hidden)).to('cpu')


        z = self.conv_z_1([X, A]) + self.conv_z_2([h_t_1, A]) + self.bias_z
        #z = self.layer_normalization_z(z)

        z = torch.sigmoid(z)

        r = self.conv_r_1([X, A]) + self.conv_r_2([h_t_1, A]) + self.bias_r
        #r = self.layer_normalization_r(r)

        r = torch.sigmoid(r)

        h_vir = torch.tanh(
            self.conv_h_1([X, A]) + r * self.conv_h_2([h_t_1, A]) + self.bias_h
        )
        #h_vir = self.layer_normalization_h(h_vir)

        h = z * h_t_1 + (1 - z) * h_vir

        if self.dropout > 0:
            h = self.dropout_layer(h)

        return h
