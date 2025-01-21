import pandas as pd
import numpy as np
from sklearn.datasets import make_classification


class MyLogReg:
    def __init__(self, n_iter=10, learning_rate=0.1, metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric_type = metric

    def __repr__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=0):
        X = X.copy()
        X.insert(0, 'bias', 1.0)
        feature_size = X.shape[1]
        self.weights = np.ones((feature_size, 1))
        y = y.to_numpy().reshape(-1, 1)

        for index in range(self.n_iter):
            y_pred = self.apply_sigmoid(X)

            loss = self.calc_loss(y_pred, y)
            grad = self.calc_grad(y_pred, y, X.to_numpy())

            self.weights -= self.learning_rate * grad
            self.metric = self.calc_metric(y, y_pred, self.metric_type)
            if verbose and index % verbose == 0:
                print(f"Iter {index} | Loss: {loss:.4f} | LR: {self.learning_rate} | metric({self.metric_type}): {self.metric:.2f}")

    def predict(self, X: pd.DataFrame):
        X_preprocessed = X.copy()
        X_preprocessed.insert(0, 'w0', 1.0)
        preds = self.apply_sigmoid(X_preprocessed)
        return preds > 0.5

    def get_best_metric(self):
        return self.metric

    def calc_metric(self, y, y_pred, metric_type):
        tp = np.sum((y == 1) & (y_pred >= 0.5))
        tn = np.sum((y == 0) & (y_pred < 0.5))
        fp = np.sum((y == 0) & (y_pred >= 0.5))
        fn = np.sum((y == 1) & (y_pred < 0.5))
        if metric_type == "accuracy":
            return (tp + tn) / (tp + tn + fp + fn)
        elif metric_type == "precision":
            return tp / (tp + fp)
        elif metric_type == "recall":
            return tp / (tp + fn)
        elif metric_type == "f1":
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            return 2 * (recall * precision) / (recall + precision)
        elif metric_type == "roc_auc":
            return 0.0
        else:
            return None

    def predict_proba(self, X: pd.DataFrame):
        X_preprocessed = X.copy()
        X_preprocessed.insert(0, 'w0', 1.0)
        preds = self.apply_sigmoid(X_preprocessed)
        return preds

    def apply_sigmoid(self, X):
        return 1 / (1 + np.exp(-X.to_numpy().dot(self.weights)))

    def calc_loss(self, y_pred, y_gt, eps=1e-8):
        return -np.mean(y_gt * np.log(y_pred + eps) + (1 - y_gt) * np.log(1 - y_pred + eps))

    def calc_grad(self, p, y_gt, X):
        return (X.T.dot(p - y_gt)) / X.shape[0]

    def get_coef(self):
        return self.weights[1:]


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


X, y = generate_dataset(n_samples=200, n_features=2, random_state=42)
logreg = MyLogReg(n_iter=100, learning_rate=0.1, metric="f1")
logreg.fit(X, y, verbose=10)
logreg_pred = logreg.predict(X)