from __future__ import annotations
from typing import Optional
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression


class RBF(object):
    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters
        self.sigmas = np.ones((self.n_clusters,))
        self.kmeans: Optional[KMeans] = None
        self.lr: Optional[LinearRegression] = None

    def phi(x, mu, sigma):
        return np.exp(- np.linalg.norm(x - mu, axis=1) ** 2 / (2 * sigma**2))

    def fit(self, X, y) -> RBF:
        self.kmeans: KMeans = KMeans(n_clusters=self.n_clusters).fit(X)
        phi = []
        # phi_j is a scalar representing the activation of the jth cluster wrt its distance with x
        # so for each cluster we need to estimate the best weights
        for center, sigma in zip(self.kmeans.cluster_centers_, self.sigmas):
            phi.append(RBF.phi(X, center, sigma))
        phi = np.array(phi)
        self.lr = LinearRegression().fit(phi.T, y)
        return self

    def predict(self, X):
        for center, sigma in zip(self.kmeans.cluster_centers_, self.sigmas):
            self.lr.predict(RBF.phi(X, center, sigma))