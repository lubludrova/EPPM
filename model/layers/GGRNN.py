import torch

from src.model.layers.GRUGRNNCell import GRUGNNCell
from src.model.layers.GRUGRNNCell import GRUGNNCell
from src.model.layers.LSTMGRNNCell import LSTMGNNCell


class GGRNN(torch.nn.Module):
    def __init__(self, n_hidden, N, F, dropout=0, return_sequences=False, bidirectional=False, n_layer=1):
        super().__init__()
        self.gru_cell = GRUGNNCell(n_hidden, N, F, dropout, n_layer=n_layer, is_last=return_sequences)
        #self.gru_cell = LSTMGNNCell(n_hidden, N, F, dropout)
        self.return_sequences = return_sequences
        self.bidirectional = bidirectional

        if self.bidirectional:
            self.bidirectional_gru_cell = GRUGNNCell(n_hidden, N, F, dropout)

    def forward(self, X):

        X_f = X[0]
        A = X[1]

        outputs = []
        hidden = None
        unbind_inputs = torch.unbind(X_f, dim=1)
        for i, x in enumerate(unbind_inputs):
            hidden = self.gru_cell([x, A], hidden)
            if not type(hidden) is tuple:
                outputs.append(hidden.clone())
            else:
                outputs.append(hidden[0].clone())

        if self.bidirectional:
            bidirectional_outputs = []
            len_input = len(unbind_inputs)
            for i in range(len_input):
                hidden = self.bidirectional_gru_cell([unbind_inputs[len_input - i - 1], A], hidden)
                bidirectional_outputs.append(hidden.clone())

            for i in range(len_input):
                outputs[i] += bidirectional_outputs[i]

        if self.return_sequences:
            return torch.stack(outputs, dim=1)
        else:
            return outputs[-1]
