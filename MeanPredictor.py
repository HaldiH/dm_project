import numpy as np

class MeanPredictor:
    def __init__(self):
        self.X_train = []
        self.y_train = []
        self.means = []

    def fit(self, X_train, y_train):
        self.means = np.mean(y_train, axis=0)
        return self

    def predict(self, X_test):
        y_pred = np.ndarray(shape=(X_test.shape[0], len(self.means)))
        for j in range(len(self.means)):
            y_pred[:, j] = self.means[j]
        return y_pred 
