import numpy as np
from scipy import sparse as sp
import torch

class TorchGenerator(torch.utils.data.Dataset):
    def __init__(self, X_prefixes, y_prefixes, vectorizer, adjacency_matrix, batch_size=32):
        self.X_prefixes = X_prefixes
        self.y_prefixes = y_prefixes
        self.vectorizer = vectorizer
        self.batch_size = batch_size
        #self.adjacency_matrix = adjacency_matrix
        #self.adjacency_matrix = TorchGenerator.normalized_adjacency(adjacency_matrix, symmetric=False)
        self.adjacency_matrix = TorchGenerator.localpooling_filter(adjacency_matrix, symmetric=False)

    def __len__(self):
        return len(self.X_prefixes)

    def collate_fn(self, data):
        X = [item[0] for item in data]
        y = [item[1] for item in data]
        last_places_activated = [item[2] for item in data]
        X_attributes = [item[3] for item in data]

        max_batch_length = 0
        for item in X:
            if len(item) > max_batch_length:
                max_batch_length = len(item)

        X_np = []
        for item in X:
            # TODO: use https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
            #for i in range(max_batch_length - len(item)):
            item.insert(0, np.zeros(shape=(max_batch_length - len(item), self.vectorizer.N, self.vectorizer.F), dtype="float32"))
            c = np.concatenate(item, axis=0)
            X_np.append(np.expand_dims(c, axis=0))

        X_attr = []
        for item in X_attributes:
            zeros = np.zeros(shape=(max_batch_length - len(item), len(X_attributes[0][0])))
            to_conct = [zeros, item]
            c = np.concatenate(to_conct, axis=0)
            X_attr.append(np.expand_dims(c, axis=0))

        X_np = np.concatenate(X_np, axis=0)
        X_attributes = np.concatenate(X_attr, axis=0)

        adj = np.expand_dims(self.adjacency_matrix, axis=0)
        conct = adj
        for i in range(len(X)-1):
            conct = np.concatenate([conct, adj], axis=0)

        Y = [y]
        Y = np.array(Y)

        return [X_np, conct, X_attributes], Y, max_batch_length, last_places_activated

    def __getitem__(self, idx):
        X_prefix, y_prefix = self.X_prefixes[idx], self.y_prefixes[idx]
        X, y_np, _, _, _, y_next_timestamp, y_attributes, last_place_activated, X_attributes = self.vectorizer.vectorize_batch([X_prefix], [y_prefix])
        # print(f"[Generator] y = {y_np[0]}")
        return X[0], y_np[0], last_place_activated, X_attributes[0]

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
                fltr[i] = TorchGenerator.normalized_adjacency(A_tilde, symmetric=symmetric)
        else:
            A_tilde = A + I
            #A_tilde = A
            fltr = TorchGenerator.normalized_adjacency(A_tilde, symmetric=symmetric)

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
            normalized_D = TorchGenerator.degree_power(A, -0.5)
            output = normalized_D.dot(A).dot(normalized_D)
        else:
            normalized_D = TorchGenerator.degree_power(A, -1.)
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
