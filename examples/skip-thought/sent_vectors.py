# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
