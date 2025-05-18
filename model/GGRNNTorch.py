import torch
import numpy as np

from src.model.graph_layers.Pooling import MaxPool, SumPool
from src.model.layers.GGRNN import GGRNN
from src.model.lstm_head import LSTMHead







class GGRNNTorch(torch.nn.Module):

    def __init__(self, graph_embedder, hyperparams, N, F, unique_activities, log_name,
                 max_len, vectorizer, attribute_count, is_search):
        super().__init__()
        self.graph_embedder = graph_embedder
        self.N = N
        self.F = F
        self.unique_activities = unique_activities
        self.log_name = log_name
        # self.max_len = max_len
        self.vectorizer = vectorizer
        self.is_search = is_search
        self.attribute_count = attribute_count
        self.attribute_count = len(list(attribute_count.keys()))
        self.attributes = attribute_count
        self.max_len = max_len

        self.rnn_1_hidden_size = hyperparams["rnn_1"]
        self.rnn_2_hidden_size = hyperparams["rnn_2"]
        self.attribute_rnn = hyperparams["attribute_rnn"]
        self.dropout_rnn_attribute = hyperparams["dropout_rnn_attribute"]
        # self.rnn_2_hidden_size = hyperparams["rnn_2"]

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.embedding_size = hyperparams["embedding"]
        self.n_layer_attribute = hyperparams["n_layer_attribute"]

        n_embeddings = 2
        self.embedded_F = n_embeddings * self.embedding_size

        self.GGRNN_1 = GGRNN(self.rnn_1_hidden_size, self.N, self.embedded_F, dropout=0, return_sequences=True,
                             bidirectional=False, n_layer=1).to(self.device)
        self.GGRNN_2 = GGRNN(self.rnn_2_hidden_size, self.N, self.rnn_1_hidden_size, dropout=0, return_sequences=True,
                             bidirectional=False, n_layer=1).to(self.device)
        #self.GGRNN_3 = GGRNN(self.rnn_1_hidden_size, self.N, self.rnn_1_hidden_size, dropout=self.dropout, return_sequences=False, n_layer=2).to(self.device)

        """
        self.gnn_hidden_size = 64
        self.graph_conv_tim = GraphConv(self.gnn_hidden_size, N, self.rnn_1_hidden_size).to(self.device)
        self.attribute_graph_convs = []
        if self.attribute_count > 0:
            for i in range(self.attribute_count):
                self.attribute_graph_convs.append(GraphConv(self.gnn_hidden_size, N, self.rnn_1_hidden_size).to(self.device))
        """

        self.global_sum_pool = SumPool()
        self.global_max_pool = MaxPool()
        emb = self.embedding_size  # hyperparams["embedding"]
        attr_emb_dim = emb * (3 + self.graph_embedder.attribute_number)
        in_dim_total = self.rnn_2_hidden_size + attr_emb_dim
        self.head = LSTMHead(in_dim=in_dim_total,  # то, что выходит из self.GGRNN_2
                             hidden=self.attribute_rnn,
                             num_classes=len(self.unique_activities) + 1,
                             num_layers=self.n_layer_attribute,
                             dropout=self.dropout_rnn_attribute).to(self.device)

        # self.dense_act = torch.nn.Linear(self.attribute_rnn, len(self.unique_activities) + 1).to(self.device)

        # static_features = 0
        # self.lstm_attributes = torch.nn.LSTM((3 + self.attribute_count) * self.embedding_size + self.rnn_2_hidden_size + static_features, self.attribute_rnn, batch_first=True, num_layers=self.n_layer_attribute, dropout=self.dropout_rnn_attribute).to(self.device)
        # #self.lstm_attributes = torch.nn.TransformerDecoderLayer(d_model = 4 * self.embedding_size + self.rnn_1_hidden_size, nhead=4, dim_feedforward=128)
        # self.layer_norm = torch.nn.LayerNorm(128)

    def forward(self, concatenation, A_in, last_places_activated, multiple_outputs, X_attributes):

        X_attributes = torch.from_numpy(X_attributes.astype(np.float32)).to(self.device)
        # X_attributes = X_attributes.float().to(self.device)

        concatenation = concatenation.permute(0, 3, 1, 2)

        concatenation = torch.nn.ZeroPad2d((0, 0, self.max_len - concatenation.shape[2], 0))(concatenation)

        concatenation = concatenation.permute(0, 2, 3, 1)

        X = self.GGRNN_1([concatenation, A_in])

        X = self.GGRNN_2([X, A_in])

        outputs = []

        # Perform the pooling
        """
        X_1 = []
        for last_activations, mini_batch in zip(last_places_activated, torch.unbind(X, dim=0)):
            if len(last_activations) > 1:
                to_sum = torch.index_select(mini_batch, index=torch.tensor(last_activations).to(self.device), dim=0)
                summed = torch.sum(to_sum, dim=0)
                X_1.append(torch.unsqueeze(summed, dim=0))
            else:
                X_1.append(torch.unsqueeze(mini_batch[last_activations[0], :], dim=0))

        X_1 = torch.cat(X_1, dim=0)
        """

        # Perform the pooling
        mini_batch_pooled = []
        for activation_list, mini_batch in zip(last_places_activated, torch.unbind(X, dim=0)):
            pooled_trace = []
            for activation_idx, graph_element in enumerate(reversed(torch.unbind(mini_batch, dim=0)), start=1):
                """
                if len(activation_list) >= activation_idx:
                    places_activated = activation_list[-activation_idx]
                    if len(places_activated) <= 1:
                        pooled_trace.insert(0, graph_element[places_activated, :])
                    else:
                        selected_indexes = torch.index_select(graph_element, dim=0, index=torch.LongTensor(places_activated).to(self.device))
                        pooled_trace.insert(0, torch.unsqueeze(torch.sum(selected_indexes, dim=0), dim=0))
                else:
                    pooled_trace.insert(0, torch.unsqueeze(torch.max(graph_element, dim=0)[0], dim=0))
                    """

                pooled_trace.insert(0, torch.unsqueeze(torch.max(graph_element, dim=0)[0], dim=0))
                """
                if len(activation_list) >= activation_idx:
                    pooled_trace.insert(0, torch.unsqueeze(torch.max(graph_element, dim=0)[0], dim=0))
                else:
                    pooled_trace.insert(0, torch.zeros(1, graph_element.shape[-1], device=self.device))
                """


            pooled_trace = torch.cat(pooled_trace, dim=0)

            mini_batch_pooled.append(torch.unsqueeze(pooled_trace, dim=0))

        X = torch.cat(mini_batch_pooled, dim=0)

        zeros = torch.zeros(X_attributes.shape[0], self.max_len - X_attributes.shape[1], X_attributes.shape[2]).to(self.device)

        X_attributes = torch.cat([zeros, X_attributes], dim=1)

        activity_embedding = self.graph_embedder.act_embedding(X_attributes[:, :, 0].long())
        time_1_embedding = self.graph_embedder.time_1_embedding(X_attributes[:, :, 1].long())
        time_2_embedding = self.graph_embedder.time_2_embedding(X_attributes[:, :, 2].long())
        attribute_embeddings = [activity_embedding, time_1_embedding, time_2_embedding]
        if self.graph_embedder.attribute_number > 0:
            for i in range(self.graph_embedder.attribute_number):
                attribute_embeddings.append(self.graph_embedder.attribute_embeddings[i](X_attributes[:, :, 3 + i].long()))

        #raw_time = X_attributes[:, :, -11:].float()
        #attribute_embeddings.append(raw_time)

        attribute_embeddings = torch.cat(attribute_embeddings, dim=-1)
        input_to_lstm = torch.cat([X, attribute_embeddings], dim=-1)
        logits = self.head(input_to_lstm)
        outputs.append(logits)

        # X_1 = self.lstm_attributes(input_to_lstm)[0]
        # X_1 = X_1[:, -1, :]
        # X_1 = self.dense_act(X_1)
        # outputs.append(X_1)

        if multiple_outputs:
            return outputs
        else:
            return outputs[0]


    @torch.no_grad()
    def graph_embedding(self, concat):
        """
        concat : тензор, вернувшийся из GraphEmbedder  [B,T,N,F]
        Берём последнюю временную проекцию так же,
        как это делается перед LSTM внутри forward.
        Возвращает [B, rnn_2_hidden_size].
        """
        return concat[:, -1, :, :].max(dim=1).values   # exactly X (см. ваш код)

    @torch.no_grad()
    def build_attr_seq(self, X_attributes, max_batch_length):
        """
        Воспроизводит pipeline формирования attr_seq (embeddings + cat)
        и возвращает тензор [B,T,a_dim] – полностью готовый вход
        для LSTMHead.
        """
        zeros = torch.zeros((X_attributes.size(0),
                             max_batch_length - X_attributes.size(1),
                             X_attributes.size(2)), device=X_attributes.device)
        X_attr = torch.cat([zeros, X_attributes], dim=1)   # дополняем до одного размера

        act  = self.graph_embedder.act_embedding(X_attr[:, :, 0].long())
        t1   = self.graph_embedder.time_1_embedding(X_attr[:, :, 1].long())
        t2   = self.graph_embedder.time_2_embedding(X_attr[:, :, 2].long())
        emb  = [act, t1, t2]

        if self.graph_embedder.attribute_number > 0:
            for i in range(self.graph_embedder.attribute_number):
                emb.append(self.graph_embedder.attribute_embeddings[i](
                            X_attr[:, :, 3+i].long()))
        return torch.cat(emb, dim=-1)         # [B,T,a_dim]