import torch

class GraphEmbedder(torch.nn.Module):
    def __init__(self, hyperparams, N, F, unique_activities, log_name,
                 max_len, vectorizer, attribute_count, n_transitions, n_nodes, n_bucket_time_1, n_bucket_time_2, is_search):
        super().__init__()
        self.N = N
        self.F = F
        self.unique_activities = unique_activities
        self.log_name = log_name
        #self.max_len = max_len
        self.vectorizer = vectorizer
        self.is_search = is_search
        # self.args = args
        self.attribute_count = attribute_count
        self.attribute_number = len(list(attribute_count.keys()))

        # print("Attribute count: ", attribute_count)

        self.attributes = attribute_count

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # print('GraphEmbedder', self.device)

        self.embedding_size = hyperparams["embedding"]

        self.act_embedding = torch.nn.Embedding(len(self.unique_activities) + 3, self.embedding_size, padding_idx=0).to(self.device)
        self.node_embedding = torch.nn.Embedding(n_nodes + 2, self.embedding_size, padding_idx=0).to(self.device)
        self.time_1_embedding = torch.nn.Embedding(hyperparams["n_bucket_time_1"]+ 2, self.embedding_size, padding_idx=0).to(self.device)
        self.time_2_embedding = torch.nn.Embedding(hyperparams["n_bucket_time_2"] + 2, self.embedding_size, padding_idx=0).to(self.device)
        self.transition_embedding = torch.nn.Embedding(n_transitions + 2, self.embedding_size, padding_idx=0).to(self.device)

        # print("N activities: ", len(self.unique_activities))
        # print("N nodes: ", n_nodes)
        # print("N bucket time 1: ", n_bucket_time_1)
        # print("N bucket time 2: ", n_bucket_time_2)
        # print("Transition embedding: ", n_transitions)

        self.attribute_embeddings = []
        sorted_attribute_keys = sorted(list(self.attribute_count.keys()))
        if self.attribute_number > 0:
            for i in range(self.attribute_number):
                n_attributes = self.attribute_count[sorted_attribute_keys[i]]
                # print("N attributes: ", n_attributes)
                self.attribute_embeddings.append(
                    torch.nn.Embedding(
                        n_attributes + 2, self.embedding_size, padding_idx=0).to(self.device)
                )

        # We need to do this assigment so as to be able to save the embedding weights.
        self.embedding_list = torch.nn.ModuleList(self.attribute_embeddings)


    def forward(self, input_X, input_A, max_batch_length, multiple_outputs=True):
        if type(input_X) is not torch.Tensor:
            X = torch.from_numpy(input_X).to(self.device)
            A_in = torch.from_numpy(input_A).float().to(self.device)
        else:
            X = input_X.to(self.device)
            A_in = input_A.float().to(self.device)

        node_id = torch.reshape(X[:, :, :, 1], (-1, max_batch_length*self.N)).long()
        transition_id = torch.reshape(X[:, :, :, 4], (-1, max_batch_length*self.N)).long()

        #if self.attribute_count > 0:
        #    rest = X[:, :, :, 2:-self.attribute_count]
        #else:
        #    rest = X[:, :, :, 2:]

        node_embedding = self.node_embedding(node_id)
        transition_embedding = self.transition_embedding(transition_id)

        node_embedding = torch.reshape(node_embedding, (-1, max_batch_length, self.N, self.embedding_size))
        transition_embedding = torch.reshape(transition_embedding, (-1, max_batch_length, self.N, self.embedding_size))

        concatenation_list = [node_embedding, transition_embedding]

        concatenation = torch.cat(concatenation_list, dim=-1)

        return concatenation, A_in
