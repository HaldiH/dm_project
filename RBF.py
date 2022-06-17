from __future__ import annotations
from typing import Optional
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np


class RBF(object):
    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters
        self.sigmas = np.ones((self.n_clusters,))
        self.kmeans: Optional[KMeans] = None
        self.lr: Optional[LinearRegression] = None

    def __phi(x, mu, sigma):
        return np.exp(- np.linalg.norm(x - mu, axis=1) ** 2 / (2 * sigma**2))

    def get_phi(self, x):
        return np.array([RBF.__phi(x, center, sigma)
                         for center, sigma in zip(self.kmeans.cluster_centers_, self.sigmas)]).T

    def fit(self, X, y) -> RBF:
        self.kmeans: KMeans = KMeans(n_clusters=self.n_clusters).fit(X)
        # phi_j is a scalar representing the activation of the jth cluster wrt its distance with x
        # so for each cluster we need to estimate the best weights
        phi = self.get_phi(X)
        self.lr = LinearRegression().fit(phi, y)
        return self

    def predict(self, X):
        phi = self.get_phi(X)
        return self.lr.predict(phi)
