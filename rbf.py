from __future__ import annotations
from locale import normalize
from typing import List, Optional
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import StandardScaler


class RBF(object):
    def __init__(self, n_clusters=8, sigma=1.0, normalize=False):
        self.n_clusters = n_clusters
        # self.sigmas: Optional[List[np.ndarray]] = None
        self.sigmas = sigma * np.ones((n_clusters,))
        self.kmeans: Optional[KMeans] = None
        self.lr: Optional[LinearRegression] = None
        self.normalize = normalize
        if normalize:
            self.features_scaler = StandardScaler()
            self.targets_scaler = StandardScaler()

    def phi(x, mu, sigma: np.ndarray):
        return np.exp(- np.linalg.norm(x - mu, axis=-1)**2 / (2 * sigma**2))
        # If we use a covariance matrix:
        # return np.exp(-1/2 * np.dot(np.dot(x - mu, np.linalg.pinv(sigma)), x - mu))

    def __phi(self, X):
        return np.array(
            [RBF.phi(X, center, sigma)
             for center, sigma in zip(self.kmeans.cluster_centers_, self.sigmas)]
        ).T
        # return np.array(
        #     [np.apply_along_axis(lambda x: RBF.phi(x, center, sigma), 1, X)
        #      for center, sigma in zip(self.kmeans.cluster_centers_, self.sigmas)]).T

    def fit(self, X, y) -> RBF:
        if self.normalize:
            X = self.features_scaler.fit_transform(X)
            y = self.targets_scaler.fit_transform(y)
        self.kmeans: KMeans = KMeans(n_clusters=self.n_clusters)
        self.kmeans.fit(X)
        # self.sigmas = [np.cov(X[pred == cluster].T)
        #                for cluster in range(self.n_clusters)]
        # phi_j is a scalar representing the activation of the jth cluster wrt its distance with x
        # so for each cluster we need to estimate the best weights
        phi = self.__phi(X)
        self.lr = LinearRegression().fit(phi, y)
        return self

    def predict(self, X, denormalize=True):
        if self.normalize:
            X = self.features_scaler.transform(X)
        phi = self.__phi(X)
        y_pred = self.lr.predict(phi)
        if self.normalize and denormalize:
            y_pred = self.targets_scaler.inverse_transform(y_pred)
        return y_pred
