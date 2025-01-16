import pandas as pd
import numpy as np
import random
from sklearn.datasets import make_classification


class MyLogReg:
    def __init__(self, n_iter=10, learning_rate=0.1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None

    def __repr__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=0):
        X.insert(0, '', 1.0)
        feature_size = X.shape[1]
        self.weights = np.ones((feature_size, 1))  # init
        for i in range(self.n_iter):
            # predict
            y_pred = X.dot(self.weights)

            # evaluate loss and grad
            loss = self.calc_loss(y_pred, y)
            grad = self.calc_grad(y_pred, y)

            # update step
            self.weights = self.weights - self.learning_rate * grad
            if verbose and index % verbose == 0:
                basic_log = f"{'start' if index == 0 else index} | loss: {loss} | lr: {current_learning_rate}"
                print(basic_log)


    def calc_loss(self, y_pred, y_gt, eps=1e-8):
        p = 1 / (1 + np.exp(-y_pred))
        return -np.mean(y_gt * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    def calc_grad(self, y_pred, y_gt):
        return 1 / y_pred.shape[0] * (y_pred.values - y_gt.values.reshape(-1, 1)).T.dot(X.values)

    def get_coef(self):
        return self.weights


def generate_dataset(n_samples=200, n_features=3, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_classes=2,
        random_state=random_state
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    y = pd.Series(y, name="target")
    return X, y

X, y = generate_dataset(n_samples=200, n_features=3, random_state=42)
logreg = MyLogReg()
logreg.fit(X, y)
print(logreg.get_coeffs())