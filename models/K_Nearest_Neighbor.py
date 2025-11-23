# models/KNN.py
import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

class KNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def predict(self, X):
        X = np.array(X)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        # Tính khoảng cách Euclidean
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        # Lấy k hàng xóm gần nhất
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        # Trả về nhãn phổ biến nhất
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict_proba(self, X):
        X = np.array(X)
        probas = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            classes, counts = np.unique(k_labels, return_counts=True)
            proba = np.zeros(len(np.unique(self.y_train)))
            for cls, count in zip(classes, counts):
                proba[int(cls)] = count / self.k
            probas.append(proba)
        return np.array(probas)