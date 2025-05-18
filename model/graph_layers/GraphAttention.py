import torch


class GraphAttention(torch.nn.Module):
    def __init__(self, channels, N, F, attn_heads=4):
        super().__init__()
        self.N = N
        self.channels = channels
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # print('GraphAttention', self.device)
        self.kernel = torch.nn.Parameter(data=torch.Tensor(F, attn_heads, channels), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.kernel)

        self.attn_kernel_self = torch.nn.Parameter(data=torch.Tensor(channels, attn_heads, 1), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.attn_kernel_self)

        self.attn_kernel_neighs = torch.nn.Parameter(data=torch.Tensor(channels, attn_heads, 1), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.attn_kernel_neighs)

        self.dropout = torch.nn.Dropout(0.2)

        #self.bias = torch.nn.Parameter(data=torch.Tensor(N, channels), requires_grad=True)
        #torch.nn.init.ones_(self.bias)

        #self.linear = torch.nn.Linear(F, channels)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        X = x[0]
        A = x[1]

        for i in range(A.shape[1]):
            A[:, i, i] = 1

        if len(X.shape) == 2:
            X = X.unsqueeze(dim=0)
            X = X.repeat(A.shape[0], 0, 0)

        X = torch.einsum("...ni, iho -> ...nho", X, self.kernel)
        attn_for_self = torch.einsum("...nhi, iho -> ...nho", X, self.attn_kernel_self)
        attn_for_neighbors = torch.einsum("...nhi, iho -> ...nho", X, self.attn_kernel_neighs)
        attn_for_neighbors = torch.einsum("...abc -> ...cba", attn_for_neighbors)

        attn_coef = attn_for_self + attn_for_neighbors
        attn_coef = torch.nn.LeakyReLU(0.2)(attn_coef)

        mask = -10e9 * (1.0 - A)
        attn_coef += mask[..., None, :]
        attn_coef = torch.nn.Softmax(dim=-1)(attn_coef)
        attn_coef_drop = self.dropout(attn_coef)

        output = torch.einsum("...nhm, ...mhi -> ...nhi", attn_coef_drop, X)

        output = torch.mean(output, dim=-2)


        return output
