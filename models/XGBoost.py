# models/XGBoost.py
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class XGBoostModel(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def __repr__(self):
        return f"XGBoost(n_est={self.n_estimators}, depth={self.max_depth}, lr={self.learning_rate})"