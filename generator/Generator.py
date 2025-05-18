import tensorflow as tf
import numpy as np
import spektral

class Generator(tf.keras.utils.Sequence):
    def __init__(self, X_prefixes, y_prefixes, vectorizer, log_file, adjacency_matrix, batch_size=32):
        self.X_prefixes = X_prefixes
        self.y_prefixes = y_prefixes
        self.vectorizer = vectorizer
        self.log_file = log_file
        self.batch_size = batch_size
        self.adjacency_matrix = adjacency_matrix
        #self.adjacency_matrix = spektral.utils.convolution.localpooling_filter(adjacency_matrix, symmetric=False)

    def __len__(self):
        return int(np.floor(len(self.X_prefixes) / self.batch_size))

    def __getitem__(self, index):
        X = self.X_prefixes[index*self.batch_size:(index+1)*self.batch_size]
        y = self.y_prefixes[index*self.batch_size:(index+1)*self.batch_size]

        X_np, y_np, _, _, _ = self.vectorizer.vectorize_batch(X, y)

        adj = np.expand_dims(self.adjacency_matrix, axis=0)
        conct = adj
        for i in range(self.batch_size-1):
            conct = np.concatenate([conct, adj], axis=0)

        return [X_np, conct], y_np

