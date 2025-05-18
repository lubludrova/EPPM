import math
import torch


class GCNII(torch.nn.Module):
    """
    Implementation of: https://arxiv.org/pdf/2007.02133.pdf
    Adaptation of: https://github.com/chennnM/GCNII
    """
    def __init__(self, channels, N, F, n_layer, alpha=0.2, l_lambda=0.8, is_last=True):
        super().__init__()
        self.N = N
        self.channels = channels
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # print('GCNII', self.device)
        self.is_last = is_last

        self.kernel = torch.nn.Parameter(data=torch.Tensor(channels, channels), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.kernel)

        self.kernel_projection = torch.nn.Parameter(data=torch.Tensor(F, channels), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.kernel_projection)

        #torch.nn.init.kaiming_uniform_(self.kernel)

        self.bias = torch.nn.Parameter(data=torch.Tensor(N, channels), requires_grad=True)
        torch.nn.init.ones_(self.bias)

        self.alpha = alpha
        self.l_lambda = l_lambda
        self.n_layer = n_layer

        #self.linear = torch.nn.Linear(F, channels)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        X = x[0]
        A = x[1]

        X_h = X
        X_0 = X

        beta = math.log(self.l_lambda / self.n_layer + 1)
        PX = torch.matmul(A, X_h)
        # The projection op is needed since the original code performs a linear transformation
        # Of the starting inputs
        # TODO: disable the projection operation in deeper layers of the network (RNN)
        first = (1 - self.alpha) * PX + self.alpha * X_0
        if self.n_layer == 1:
            first = torch.matmul(first, self.kernel_projection)

        output = beta * torch.matmul(first, self.kernel) + (1 - beta) * first

        output += self.bias

        #out = torch.matmul(X, self.kernel)
        #out = torch.matmul(A, out)

        #out2 = torch.matmul(X, self.kernel_2)


        return output
