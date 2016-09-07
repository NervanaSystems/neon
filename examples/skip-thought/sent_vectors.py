# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
# ----------------------------------------------------------------------------

import numpy as np


class SentenceVector(object):

    """
    A container class of sentence vectors for easy query similar sentences etc.
    """

    def __init__(self, vectors, text):
        """
        Initialize a SentenceVectors class object

        Arguments:
            vectors (ndarray, (#sentences, vector dimension)): sentence vectors
            text (list, #sentences): sentence texts
        """
        self.vectors = vectors
        self.text = text

        if isinstance(self.text, list):
            assert self.vectors.shape[0] == len(self.text)
        elif isinstance(self.text, np.ndarray):
            assert self.vectors.shape[0] == self.text.shape[0]

        norms = np.linalg.norm(self.vectors, axis=1)
        self.vectors = self.vectors / norms.reshape(-1, 1)

    def find_similar_idx(self, query, n=10):
        """
        Find similar sentences by vector distances
        metric = dot(vectors_of_vectors, vectors_of_target_vector)
        Uses a precomputed vectors of the vectors
        Parameters

        Arguments:
            query (ndarray): query sentence vector
            n (int): top n number of neighbors

        Returns:
            position in self.vocab_w2id
            cosine similarity
        """
        query = query / np.linalg.norm(query)
        metrics = np.dot(self.vectors, query.T)

        best = np.argsort(metrics.ravel())[::-1][:n]
        best_metrics = metrics[best]

        return best, best_metrics

    def find_similar(self, query, n=10):
        """
        Find similar sentences by vector distances
        metric = dot(vectors_of_vectors, vectors_of_target_vector)
        Uses a precomputed vectors of the vectors
        Parameters

        Arguments:
            query (ndarray): query sentence vector
            n (int): top n number of neighbors

        Returns:
            position in self.vocab_w2id
            cosine similarity
        """
        query = query / np.linalg.norm(query)
        metrics = np.dot(self.vectors, query.T)

        best = np.argsort(metrics.ravel())[::-1][:n]
        best_metrics = metrics[best]
        nearest = [self.text[b] for b in best.tolist()]

        return nearest, best_metrics


    def find_similar_with_idx(self, idx, n=10):
        if isinstance(idx, list):
            best = []
            for i in idx:
                best += self.find_similar_with_idx(i, n)
            return best
        else:
            query = self.vectors[idx]
            metrics = np.dot(self.vectors, query.T)

            best = np.argsort(metrics.ravel())[::-1][:n]
            return best.tolist()
