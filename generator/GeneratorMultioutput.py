import numpy as np
from scipy import sparse as sp

class GeneratorMultioutput:
    def __init__(self, X_prefixes, y_prefixes, vectorizer, adjacency_matrix, batch_size=32):
        self.X_prefixes = X_prefixes
        self.y_prefixes = y_prefixes
        self.vectorizer = vectorizer
        self.batch_size = batch_size
        #self.adjacency_matrix = adjacency_matrix
        #self.adjacency_matrix = GeneratorMultioutput.normalized_adjacency(adjacency_matrix, symmetric=False)
        self.adjacency_matrix = GeneratorMultioutput.localpooling_filter(adjacency_matrix, symmetric=False)
        self.rng = np.random.default_rng(42)

    def __len__(self):
        return int(np.floor(len(self.X_prefixes) / self.batch_size))

    def get(self):
        n_indexes = int(np.floor(len(self.X_prefixes) / self.batch_size))
        index_list = np.arange(n_indexes)
        self.rng.shuffle(index_list)
        for index in index_list:
            X = self.X_prefixes[index*self.batch_size:(index+1)*self.batch_size]
            y = self.y_prefixes[index*self.batch_size:(index+1)*self.batch_size]

            X, y_np, _, _, _, y_next_timestamp, y_attributes, last_places_activated, X_attributes = self.vectorizer.vectorize_batch(X, y)

            #print("X generator: ", X.shape)
            max_batch_length = 0
            for item in X:
                #print("Item shape: ", len(item))
                if len(item) > max_batch_length:
                    max_batch_length = len(item)

            batch_size_lengths = []

            #print("Max batch length: ", max_batch_length)

            X_np = []
            for item in X:
                # TODO: use https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
                for i in range(max_batch_length - len(item)):
                    item.insert(0, np.zeros(shape=(1, self.vectorizer.N, self.vectorizer.F), dtype="float32"))
                c = np.concatenate(item, axis=0)
                X_np.append(np.expand_dims(c, axis=0))

            X_np = np.concatenate(X_np, axis=0)

            X_attr = []
            for item in X_attributes:
                zeros = np.zeros(shape=(max_batch_length - len(item), len(X_attributes[0][0])))
                to_conct = [zeros, item]
                c = np.concatenate(to_conct, axis=0)
                X_attr.append(np.expand_dims(c, axis=0))
            X_attributes = np.concatenate(X_attr, axis=0)

            #print("X_np shape: ", X_np.shape)

            adj = np.expand_dims(self.adjacency_matrix, axis=0)
            conct = adj
            for i in range(self.batch_size-1):
                conct = np.concatenate([conct, adj], axis=0)

            Y = [y_np, y_next_timestamp]
            for attr in y_attributes:
                Y.append(attr)

            yield [X_np, conct, X_attributes], Y, max_batch_length, [last_places_activated]


    @staticmethod
    def localpooling_filter(A, symmetric=True):
        r"""
        Computes the graph filter described in
        [Kipf & Welling (2017)](https://arxiv.org/abs/1609.02907).
        :param A: array or sparse matrix with rank 2 or 3;
        :param symmetric: boolean, whether to normalize the matrix as
        \(\D^{-\frac{1}{2}}\A\D^{-\frac{1}{2}}\) or as \(\D^{-1}\A\);
        :return: array or sparse matrix with rank 2 or 3, same as A;
        """
        fltr = A.copy()
        if sp.issparse(A):
            I = sp.eye(A.shape[-1], dtype=A.dtype)
        else:
            I = np.eye(A.shape[-1], dtype=A.dtype)
        if A.ndim == 3:
            for i in range(A.shape[0]):
                # TODO: disable self loops for now
                A_tilde = A[i] + I
                #A_tilde = A[i]
                fltr[i] = GeneratorMultioutput.normalized_adjacency(A_tilde, symmetric=symmetric)
        else:
            A_tilde = A + I
            #A_tilde = A
            fltr = GeneratorMultioutput.normalized_adjacency(A_tilde, symmetric=symmetric)

        if sp.issparse(fltr):
            fltr.sort_indices()
        return fltr

    @staticmethod
    def normalized_adjacency(A, symmetric=True):
        r"""
        Normalizes the given adjacency matrix using the degree matrix as either
        \(\D^{-1}\A\) or \(\D^{-1/2}\A\D^{-1/2}\) (symmetric normalization).
        :param A: rank 2 array or sparse matrix;
        :param symmetric: boolean, compute symmetric normalization;
        :return: the normalized adjacency matrix.
        """
        if symmetric:
            normalized_D = GeneratorMultioutput.degree_power(A, -0.5)
            output = normalized_D.dot(A).dot(normalized_D)
        else:
            normalized_D = GeneratorMultioutput.degree_power(A, -1.)
            output = normalized_D.dot(A)
        return output

    @staticmethod
    def degree_power(A, k):
        r"""
        Computes \(\D^{k}\) from the given adjacency matrix. Useful for computing
        normalised Laplacian.
        :param A: rank 2 array or sparse matrix.
        :param k: exponent to which elevate the degree matrix.
        :return: if A is a dense array, a dense array; if A is sparse, a sparse
        matrix in DIA format.
        """
        degrees = np.power(np.array(A.sum(1)), k).flatten()
        degrees[np.isinf(degrees)] = 0.
        if sp.issparse(A):
            D = sp.diags(degrees)
        else:
            D = np.diag(degrees)
        return D
