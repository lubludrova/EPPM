import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GGRNN(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Графовые слои (GCN)
        self.gcn_layers = nn.ModuleList([
            GCNConv(
                in_channels=config['embedding_dim'],
                out_channels=config['gcn_hidden_dim']
            ) for _ in range(config['num_gcn_layers'])
        ])

        # Нормализация и дропаут для графовых слоёв
        self.gcn_dropout = nn.Dropout(config['dropout'])

        # Рекуррентный блок (GRU)
        self.rnn = nn.GRU(
            input_size=config['gcn_hidden_dim'],
            hidden_size=config['rnn_hidden_dim'],
            num_layers=config['num_rnn_layers'],
            dropout=config['dropout'] if config['num_rnn_layers'] > 1 else 0,
            batch_first=True
        )

        # Выходной классификатор
        self.fc = nn.Linear(config['rnn_hidden_dim'], num_classes)

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        for layer in self.gcn_layers:
            nn.init.xavier_uniform_(layer.lin.weight)  # Правильный путь к весам
            if layer.lin.bias is not None:
                nn.init.zeros_(layer.lin.bias)

        # Инициализация для GRU
        for name, param in self.rnn.named_parameters():  # Убрать цикл по слоям!
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # Инициализация классификатора
        nn.init.kaiming_normal_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, data):
        """
        Входные данные:
            data.x: Node features [num_nodes, embedding_dim]
            data.edge_index: Графовая структура [2, num_edges]
            data.sequences: Временные последовательности [batch_size, seq_len, gcn_hidden_dim]
        """
        # Графовая часть
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)

        for gcn in self.gcn_layers:
            x = gcn(x, edge_index)
            x = F.relu(x)
            x = self.gcn_dropout(x)

        # Рекуррентная часть
        sequences = data.sequences.to(self.device)
        outputs, _ = self.rnn(sequences)  # outputs: [batch_size, seq_len, rnn_hidden_dim]
        last_output = outputs[:, -1, :]  # Последний скрытый вектор

        # Классификация
        logits = self.fc(last_output)
        return logits

    def explain(self, input_data):
        """Крючок для explainability (заглушка)"""
        # TODO: Интегрировать GNNExplainer или аналоги
        return {"status": "Explainability not implemented"}