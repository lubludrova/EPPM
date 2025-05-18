import torch


class GraphConv(torch.nn.Module):
    def __init__(self, channels, N, F):
        super().__init__()
        self.N = N
        self.channels = channels
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # print('GraphConv', self.device)
        self.kernel = torch.nn.Parameter(data=torch.Tensor(F, channels), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.kernel)
        #torch.nn.init.kaiming_uniform_(self.kernel)

        self.bias = torch.nn.Parameter(data=torch.Tensor(N, channels), requires_grad=True)
        torch.nn.init.ones_(self.bias)

        # self.linear = torch.nn.Linear(F, channels)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        X = x[0].to('cpu')
        A = x[1].to('cpu')

        out = torch.matmul(X, self.kernel)

        # out = self.linear(X)

        out = torch.matmul(A, out)

        #expand_bias = self.bias.expand(self.bias, self.N, self.channels)
        out = out + self.bias

        #output = self.relu(out)
        output = out

        return output
