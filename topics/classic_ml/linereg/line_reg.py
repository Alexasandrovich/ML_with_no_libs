import pandas as pd
import numpy as np

class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1,
                 metric=None,
                 reg=None, l1_coef=0.0, l2_coef=0.0):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric_type = metric
        self.weights = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def __repr__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=0):
        X.insert(0, '', 1.0)
        feature_size = X.shape[1]
        self.weights = np.ones((feature_size, 1))  # init
        for index in range(self.n_iter):
            y_pred = X.values @ self.weights

            # evaluate grad with regularization
            loss = self.evaluate_metric(y, y_pred, 'mse') + self.get_regularization(self.weights)
            grad = (2 / X.shape[0] * (y_pred.flatten() - y.values) @ X.values).reshape(-1, 1) + self.get_regularization_slope(self.weights)

            # update grad descent
            if not isinstance(self.learning_rate, float):
                # got lambda function -> lr calculates dynamically
                current_learning_rate = self.learning_rate(index + 1)
            else:
                current_learning_rate = self.learning_rate
            self.weights = self.weights - current_learning_rate * grad.reshape(-1, 1)
            y_pred = X.values @ self.weights

            # evaluate current metric
            self.best_score = self.evaluate_metric(y, y_pred, self.metric_type)  # todo: check bestness
            if verbose and index % verbose == 0:
                basic_log = f"{'start' if index == 0 else index} | loss: {loss} | lr: {current_learning_rate}"
                if self.metric_type:
                    print(f"{basic_log} | {self.metric_type}: {self.best_score}")
                else:
                    print(f"{basic_log}")

    def predict(self, X: pd.DataFrame):
        X.insert(0, '', 1.0)
        return X.values @ self.weights

    def get_coef(self):
        return self.weights[1:]

    def evaluate_metric(self, y_gt, y_pred, metric_type=None):
        if metric_type == 'mae':
            return np.mean(np.abs(y_gt.values - y_pred.flatten()))
        elif metric_type == 'mse':
            return np.mean((y_gt.values - y_pred.flatten()) ** 2)
        elif metric_type == 'rmse':
            return np.sqrt(np.mean((y_gt.values - y_pred.flatten()) ** 2))
        elif metric_type == 'mape':
            return 100 * np.mean(np.abs((y_gt.values - y_pred.flatten()) / y_gt.values))
        elif metric_type == 'r2':
            ss_res = np.sum((y_gt.values - y_pred.flatten()) ** 2)
            ss_tot = np.sum((y_gt.values - np.mean(y_gt.values)) ** 2)
            return 1 - (ss_res / ss_tot)
        else:
            return None

    def get_best_score(self):
        return self.best_score

    def get_regularization(self, weights):
        if self.reg == "l1":
            return np.sum(np.abs(weights)) * self.l1_coef
        elif self.reg == "l2":
            return np.sum(weights ** 2) * self.l2_coef
        elif self.reg == "elasticnet":
            return np.sum(np.abs(weights)) * self.l1_coef + np.sum(weights ** 2) * self.l2_coef
        else:
            return 0.0

    def get_regularization_slope(self, weights):
        if self.reg == "l1":
            return np.sign(weights) * self.l1_coef
        elif self.reg == "l2":
            return 2 * weights * self.l2_coef
        elif self.reg == "elasticnet":
            return np.sign(weights) * self.l1_coef + 2 * weights * self.l2_coef
        else:
            return 0.0


MyLineReg = MyLineReg(metric='r2', reg='elasticnet', learning_rate=lambda iter: 0.5 * (0.85 ** iter))
X = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)))
Y = pd.Series(np.random.randn(100))
MyLineReg.fit(X, Y, 10)
print(MyLineReg.get_coef())
