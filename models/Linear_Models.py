# models/LinearModels.py
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class LinearModelClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper cho Logistic Regression (binary) và Softmax Regression (multiclass)
    Tự động chọn solver phù hợp
    """
    def __init__(self, penalty='l2', C=1.0, max_iter=1000, random_state=42):
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        n_classes = len(np.unique(y))
        
        if n_classes == 2:
            # Binary: Logistic Regression
            self.model = LogisticRegression(
                penalty=self.penalty,
                C=self.C,
                max_iter=self.max_iter,
                random_state=self.random_state,
                solver='lbfgs'
            )
        else:
            # Multiclass: Softmax Regression
            self.model = LogisticRegression(
                penalty=self.penalty,
                C=self.C,
                max_iter=self.max_iter,
                random_state=self.random_state,
                solver='lbfgs',
                multi_class='multinomial'
            )
        
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def __repr__(self):
        return f"LinearModel(C={self.C}, penalty={self.penalty})"