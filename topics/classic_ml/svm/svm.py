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

    def fit(self, X: pd.DataFrame, Y: pd.Series, verbose=0):
        Y = Y.to_numpy().reshape(-1, 1)
        Y[np.where(Y == 0)] = -1  # need only -1/1 labels

        feature_size = X.shape[1]
        self.weights = np.ones((feature_size, 1))
        self.b = 1
        for step in range(self.n_iter):
            for index, row in X.iterrows():
                x = row.to_numpy().reshape(-1, 1)
                y = Y[index]
                if self.is_correct_classification(x, y):
                    grad_w, grad_b = self.calc_grad(x, y, with_both_loss=False)
                else:
                    grad_w, grad_b = self.calc_grad(x, y, with_both_loss=True)

                self.weights = self.weights - self.learning_rate * grad_w
                self.b = self.b - self.learning_rate * grad_b

            loss = self.calc_loss(X, Y)
            self.print_log(step, loss, verbose)

    def is_correct_classification(self, x, y):
        return y * (self.weights.T @ x + self.b) >= 1

    def calc_grad(self, x, y, with_both_loss):
        if with_both_loss:
            grad_w, grad_b = 2 * self.weights - y * x, -y
        else:
            grad_w, grad_b = 2 * self.weights, 0.0

        return grad_w, grad_b

    def calc_loss(self, X, Y):
        reg = np.sum(self.weights ** 2)
        margin = Y.reshape(-1, 1) * (X @ self.weights + self.b)
        hinge_loss = (np.sum(np.maximum(0, 1 - margin))).to_numpy()[0]
        return reg + hinge_loss

    def print_log(self, step, loss, verbose):
        if verbose and step % verbose == 0:
            basic_log = f"{'start' if step == 0 else step} | loss: {loss}"
            print(basic_log)

    def get_coef(self):
        if isinstance(self.b, np.ndarray):
            return self.weights.ravel(), self.b[0]
        else:
            return self.weights.ravel(), self.b

def generate_test_data(n_samples=100, n_features=2):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features) * 2
    Y = np.where(X[:, 0] + X[:, 1] > 0, 1, 0)
    return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)]), pd.Series(Y)

X_test, Y_test = generate_test_data()

svm = MySVM(n_iter=10, learning_rate=0.01)
svm.fit(X_test, Y_test, verbose=1)

weights, bias = svm.get_coef()
print("Final weights:", weights)
print("Final bias:", bias)


