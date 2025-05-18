import torch

from src.model.GGRNNTorch import GGRNNTorch
from src.model.GraphEmbedder import GraphEmbedder
from src.model.lstm_head import LSTMHead





class MainModel(torch.nn.Module):
    def __init__(self, hyperparams, N, F, unique_activities, log_name,
                 max_len, vectorizer, attribute_count, n_transitions, n_nodes, n_bucket_time_1, n_bucket_time_2, architecture, is_search):
        super().__init__()

        self.graph_embedder = GraphEmbedder(hyperparams, N, F, unique_activities, log_name, max_len, vectorizer,
                                            attribute_count, n_transitions, n_nodes, n_bucket_time_1, n_bucket_time_2,
                                            is_search)

        architecture = architecture.lower()
        if architecture == "ggrnn":
            self.learning_model = GGRNNTorch(self.graph_embedder, hyperparams, N, F, unique_activities, log_name, max_len, vectorizer,
                                    attribute_count, is_search)

        else:
            print("Unrecognized architecture")
            import sys
            sys.exit(0)

        self.optimizer = hyperparams["optimizer"]
        self.lr = hyperparams["lr"]

    def forward(self, input_X, input_A, max_batch_length, last_places_activated, X_attributes, multiple_outputs=True):
        concatenation, A_in = self.graph_embedder(input_X, input_A, max_batch_length, multiple_outputs)

        return self.learning_model(concatenation, A_in, last_places_activated, multiple_outputs, X_attributes)
        #return self.ggrnn(concatenation, A_in, multiple_outputs)
