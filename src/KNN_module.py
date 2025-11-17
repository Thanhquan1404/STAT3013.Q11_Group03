"""
Usage:
    knn = KNNClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = knn.score(X_test, y_test)
Functions:
    find_optimal_k(X_train, y_train, X_test, y_test, k_range=range(1, 31), weights='distance', metric='minkowski', p=2)
    grid_search_knn(X_train, y_train, param_grid, cv=5, scoring='accuracy')
Class:
    KNNClassifier(n_neighbors=5, weights='uniform', metric='minkowski', p=2)
Methods in Class
    fit(self, X_train, y_train)
    predict(self, X)
    predict_proba(self, X)
    score(self, X, y)
    kneighbors(self, X, n_neighbors=None)
    get_params(self)
    set_params(self, **params)
    predict_with_debug(self, X, y_true=None, show_samples=10)
    _check_fitted(self)
=============================================================================
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as SKLearnKNN

class KNNClassifier:
    def __init__(self, n_neighbors=5, weights='uniform', metric='minkowski', p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        
        self.model = SKLearnKNN(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p,
            n_jobs=-1
        )
        
        self.is_fitted = False
        self.classes_ = None
    
    # ------------------------------------------------------------------
    def fit(self, X_train, y_train):
        """Train KNN model (stores training data)."""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        self.classes_ = np.unique(y_train)
        print(f"✔️ KNN trained | K={self.n_neighbors}, weights={self.weights}")
        return self
    
    # ------------------------------------------------------------------
    def predict(self, X):
        """Predict class labels."""
        self._check_fitted()
        return self.model.predict(X)
    
    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """Predict class probabilities."""
        self._check_fitted()
        return self.model.predict_proba(X)
    
    # ------------------------------------------------------------------
    def score(self, X, y):
        """Calculate accuracy score."""
        self._check_fitted()
        return self.model.score(X, y)
    
    # ------------------------------------------------------------------
    def kneighbors(self, X, n_neighbors=None):
        """
        Find K nearest neighbors.
        
        Returns:
            distances: Distance to each neighbor
            indices: Index of each neighbor in training set
        """
        self._check_fitted()
        return self.model.kneighbors(X, n_neighbors)
    
    # ------------------------------------------------------------------
    def get_params(self):
        """Get model hyperparameters (sklearn style)."""
        return {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'metric': self.metric,
            'p': self.p
        }
    
    def set_params(self, **params):
        """Set model hyperparameters (sklearn style)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.model.set_params(**params)
        return self
    
    # ------------------------------------------------------------------
    def predict_with_debug(self, X, y_true=None, show_samples=10):
        """
        Show detailed predictions with probabilities.
        Useful for debugging and understanding model behavior.
        """
        y_proba = self.predict_proba(X)
        y_pred = self.predict(X)
        
        print("\n" + "="*80)
        print("                        PREDICTION DETAILS")
        print("="*80)
        
        # Header
        header = f"{'Index':<8} {'y_true':<10} {'y_pred':<10}"
        for cls in self.classes_:
            header += f"{'P(class=' + str(cls) + ')':<15}"
        header += f"{'Correct?'}"
        print(header)
        print("-"*80)
        
        correct = 0
        for i in range(min(show_samples, len(y_pred))):
            pred = int(y_pred[i])
            
            # Build row
            if y_true is not None:
                true = int(y_true[i])
                ok = "✔️" if pred == true else "✖️"
                if pred == true:
                    correct += 1
                row = f"{i:<8} {true:<10} {pred:<10}"
            else:
                row = f"{i:<8} {'?':<10} {pred:<10}"
                ok = ""
            
            # Add probabilities
            for prob in y_proba[i]:
                row += f"{prob:<15.4f}"
            row += ok
            
            print(row)
        
        # Summary
        if y_true is not None:
            acc = correct / min(show_samples, len(y_pred))
            print("-"*80)
            print(f"Accuracy (shown): {acc:.4f} ({correct}/{min(show_samples, len(y_pred))})")
            full_acc = self.score(X, y_true)
            print(f"Accuracy (full):  {full_acc:.4f}")
            print(f"Settings → K={self.n_neighbors}, weights={self.weights}, metric={self.metric}")
        
        print("="*80 + "\n")
        
        return y_pred, y_proba
    
    # ------------------------------------------------------------------
    def _check_fitted(self):
        """Check if model is fitted."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_optimal_k(X_train, y_train, X_test, y_test, 
                   k_range=range(1, 31), weights='distance', metric='minkowski', p=2):
    scores = []
    print(f"\n🔍 Testing K values: {list(k_range)}")
    
    for k in k_range:
        knn = KNNClassifier(n_neighbors=k, weights=weights, metric=metric, p=p)
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        scores.append(score)
    
    optimal_k = list(k_range)[np.argmax(scores)]
    best_score = max(scores)
    
    print(f"✔️ Optimal K={optimal_k} with accuracy={best_score:.4f}")
    
    return optimal_k, scores


def grid_search_knn(X_train, y_train, param_grid, cv=5, scoring='accuracy'):
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    
    print(f"\n🔍 Grid search started | CV={cv}, scoring={scoring}")
    
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    base_model = SKLearnKNN()
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Create KNNClassifier with best params
    best_model = KNNClassifier(**grid_search.best_params_)
    best_model.fit(X_train, y_train)
    
    print(f"✔️ Best parameters: {grid_search.best_params_}")
    print(f"✔️ Best CV score: {grid_search.best_score_:.4f}")
    
    return best_model, grid_search.best_params_


