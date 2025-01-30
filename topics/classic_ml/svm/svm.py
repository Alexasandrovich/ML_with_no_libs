import pandas as pd
import numpy as np
import random


class MySVM:
    def __init__(self, n_iter=10, learning_rate=0.001, weights=None, b=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.b = b

    def __repr__(self):
        return f"MySVM class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=0):
        X = X.copy().to_numpy()
        y = y.to_numpy().reshape(-1, 1)
        y[np.where(y == 0)] = -1  # need only -1/1 labels

        feature_size = X.shape[1]
        self.weights = np.ones((feature_size, 1))
        self.b = 1
        for step in range(self.n_iter):
            for index, row in df.iterrows():
                pass

