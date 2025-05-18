import torch
import torch.nn as nn


class LSTMHead(nn.Module):


    def __init__(self, in_dim, hidden, num_classes, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, seq):
        # seq shape [B,T,in_dim]
        out, _ = self.lstm(seq)
        return self.fc(out[:, -1, :])
