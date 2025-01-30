import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import random

class MyLogReg:
    def __init__(self, n_iter=10, learning_rate=0.1, metric=None,
                 reg=None, l1_coef=0.0, l2_coef=0.0,
                 sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric_type = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        random.seed(random_state)

    def __repr__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=0):
        X = X.copy()
        X.insert(0, 'bias', 1.0)
        feature_size = X.shape[1]
        self.weights = np.ones((feature_size, 1))
        y = y.to_numpy().reshape(-1, 1)

        for index in range(self.n_iter):
            X_sampled, y_sampled = self.get_samples(X, y)
            y_pred = self.apply_sigmoid(X_sampled)

            loss = self.calc_loss(y_pred, y_sampled)
            grad = self.calc_grad(y_pred, y_sampled, X_sampled.to_numpy())

            # update grad descent
            if not isinstance(self.learning_rate, float):
                # got lambda function -> lr calculates dynamically
                current_learning_rate = self.learning_rate(index + 1)
            else:
                current_learning_rate = self.learning_rate
            self.weights -= current_learning_rate * grad
            y_pred = self.apply_sigmoid(X_sampled)
            self.metric = self.calc_metric(y_sampled, y_pred, self.metric_type)
            if verbose and index % verbose == 0:
                print(f"Iter {index} | Loss: {loss:.4f} | LR: {current_learning_rate} | metric({self.metric_type}): {self.metric:.2f}")

    def predict(self, X: pd.DataFrame):
        X_preprocessed = X.copy()
        X_preprocessed.insert(0, 'w0', 1.0)
        preds = self.apply_sigmoid(X_preprocessed)
        return preds > 0.5

    def get_samples(self, X, y):
        if isinstance(self.sgd_sample, float):
            samples_count = int(X.shape[0] * self.sgd_sample)
        elif isinstance(self.sgd_sample, int):
            samples_count = self.sgd_sample
        else:  # none
            samples_count = X.shape[0]

        sample_rows_idx = random.sample(range(X.shape[0]), samples_count)
        return pd.DataFrame(X.values[sample_rows_idx]), y[sample_rows_idx].reshape(-1, 1)

    def get_best_score(self):
        return self.metric

    def calc_metric(self, y, y_pred, metric_type):
        tp = np.sum((y == 1) & (y_pred >= 0.5))
        tn = np.sum((y == 0) & (y_pred < 0.5))
        fp = np.sum((y == 0) & (y_pred >= 0.5))
        fn = np.sum((y == 1) & (y_pred < 0.5))
        if metric_type == "accuracy":
            return (tp + tn) / (tp + tn + fp + fn)
        elif metric_type == "precision":
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        elif metric_type == "recall":
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric_type == "f1":
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            return 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 0.0
        elif metric_type == "roc_auc":
            y_pred_rounded = np.round(y_pred, 10)

            sorted_indices = np.argsort(-y_pred_rounded, axis=0)  # По убыванию
            y_sorted = y[sorted_indices.flatten()]
            y_pred_sorted = y_pred_rounded[sorted_indices.flatten()]

            positive_count = np.sum(y_sorted == 1)
            negative_count = np.sum(y_sorted == 0)

            if positive_count == 0 or negative_count == 0:
                return 0.0

            auc_sum = 0.0

            for i in range(len(y_sorted)):
                if y_sorted[i] == 0:
                    same_score_positive = np.sum((y_sorted == 1) & (y_pred_sorted == y_pred_sorted[i]))
                    positive_above = np.sum((y_sorted == 1) & (y_pred_sorted > y_pred_sorted[i]))
                    auc_sum += positive_above + same_score_positive / 2

            return auc_sum / (positive_count * negative_count)
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
        base_loss = -np.mean(y_gt * np.log(y_pred + eps) + (1 - y_gt) * np.log(1 - y_pred + eps))
        if self.reg == "l1":
            return base_loss + self.l1_coef * np.sum(np.abs(self.weights))
        elif self.reg == "l2":
            return base_loss + self.l2_coef * np.sum(self.weights ** 2)
        elif self.reg == "elasticnet":
            return base_loss + self.l1_coef * np.sum(np.abs(self.weights)) + self.l2_coef * np.sum(self.weights ** 2)
        else:
            return base_loss

    def calc_grad(self, p, y_gt, X):
        base_grad = (X.T.dot(p - y_gt)) / X.shape[0]
        if self.reg == "l1":
            return base_grad + self.l1_coef * np.sign(self.weights)
        elif self.reg == "l2":
            return base_grad + 2 * self.l2_coef * self.weights
        elif self.reg == "elasticnet":
            return base_grad + self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights
        else:
            return base_grad

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
logreg = MyLogReg(n_iter=100, learning_rate=0.1, metric="roc_auc",
                  reg="l1", l1_coef=0.1, l2_coef=0.1,
                  sgd_sample=5, random_state=42)
logreg.fit(X, y, verbose=10)
logreg_pred = logreg.predict(X)
