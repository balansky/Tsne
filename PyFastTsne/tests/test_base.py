from unittest import TestCase
from functools import partial

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

from PyFastTsne import PyTsne


make_blobs = partial(make_blobs, random_state=0)

def pdist(X):
    """Condensed pairwise distances, like scipy.spatial.distance.pdist()"""
    return pairwise_distances(X)[np.triu_indices(X.shape[0], 1)]

class TestPyFastTSNE(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.Xy = make_blobs(20, 100, 2, shuffle=False)


    def test_tsne(self):
        X, y = self.Xy
        tsne = PyTsne(100, 2)
        E = tsne.fit_transform(X)

        self.assertEqual(E.shape, (X.shape[0], 2))

        max_intracluster = max(pdist(E[y == 0]).max(),
                               pdist(E[y == 1]).max())
        min_intercluster = pairwise_distances(E[y == 0],
                                              E[y == 1]).min()

        self.assertGreater(min_intercluster, max_intracluster)

    def test_benchmark(self):
        for N in [1000, 10000, 100000]:
            for D in [10, 100, 1000, 10000]:

                print ('=====================')
                print ('N: %d, D: %d' % (N, D))
                print ('=====================')
                X = np.random.rand(N, D)
                tsne = PyTsne(D, 2)
                mnist_tsne = tsne.fit_transform(X)