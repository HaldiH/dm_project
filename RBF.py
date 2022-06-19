from __future__ import annotations
from typing import List, Optional
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np


class RBF(object):
    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters
        self.sigmas: Optional[List[np.ndarray]] = None
        self.kmeans: Optional[KMeans] = None
        self.lr: Optional[LinearRegression] = None

    def phi(x, mu, sigma: np.ndarray):
        print(np.shape(x-mu))
        print(np.shape(sigma))
        return np.exp(-1/2 * (x - mu).T @ np.linalg.inv(sigma) @ (x - mu))

    def __phi(self, x):
        return np.array([RBF.phi(x, center, sigma) for center, sigma in zip(self.kmeans.cluster_centers_, self.sigmas)])

    def fit(self, X, y) -> RBF:
        self.kmeans: KMeans = KMeans(n_clusters=self.n_clusters).fit(X)
        pred = self.kmeans.predict(X)
        print(X[pred == 0].shape)
        self.sigmas = [np.cov(X[pred == cluster])
                       for cluster in range(self.n_clusters)]
        # phi_j is a scalar representing the activation of the jth cluster wrt its distance with x
        # so for each cluster we need to estimate the best weights
        phi = self.__phi(X)
        self.lr = LinearRegression().fit(phi, y)
        return self

    def predict(self, X):
        phi = self.phi(X)
        return self.lr.predict(phi)
