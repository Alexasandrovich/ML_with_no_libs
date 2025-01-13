import pandas as pd
import numpy as np

class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None

    def __repr__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=0):
        X.insert(0, '', 1.0)
        feature_size = X.shape[1]
        self.weights = np.ones((feature_size, 1))  # init
        for index in range(self.n_iter):
            y_pred = X.values @ self.weights
            mse = np.mean((y.values - y_pred.flatten())**2)
            grad = 2 / X.shape[0] * (y_pred.flatten() - y.values) @ X.values
            self.weights = self.weights - self.learning_rate * grad.reshape(-1, 1)
            if verbose and index % verbose == 0:
                print(f"{'start' if index == 0 else index} | loss: {mse}")

    def predict(self, X: pd.DataFrame):
        X.insert(0, '', 1.0)
        return X.values @ self.weights

    def get_coef(self):
        return self.weights[1:]



MyLineReg = MyLineReg()
X = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)))
Y = pd.Series(np.random.randn(100))
MyLineReg.fit(X, Y, 10)
print(MyLineReg.get_coef())
