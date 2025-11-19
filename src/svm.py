import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings

# Disable non-critical warnings to maintain a clean execution environment
warnings.filterwarnings('ignore')


# ====================================================================
# GENERAL SVM CLASSIFIER (NO PRINT STATEMENTS)
# ====================================================================

class SVMClassifier:
    """
    A lightweight wrapper around sklearn.svm.SVC intended for modular,
    silent operation. This class adheres to the interface conventions
    commonly used in scikit-learn, thereby ensuring compatibility with
    standardized workflows such as GridSearchCV and pipeline integration.

    The kernel type is passed directly through the `kernel` argument.
    All hyperparameters of sklearn's SVC (e.g., C, gamma, degree) are
    supported through __init__, get_params, and set_params.

    Note:
        The option `probability=True` is enforced to ensure that the
        method `predict_proba()` is always available, regardless of
        kernel choice.
    """

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale',
                 random_state=None, **kwargs):
        """
        Initialize the SVM classifier with user-specified hyperparameters.

        Parameters:
            C (float): Regularization strength.
            kernel (str): Kernel function to be applied.
            degree (int): Degree of polynomial kernel (if applicable).
            gamma (str or float): Kernel coefficient.
            random_state (int): Seed for reproducibility.
            **kwargs: Additional parameters forwarded to sklearn.svm.SVC.

        Notes:
            Probability estimates are always enabled. Although this incurs
            additional computational cost during training, it ensures that
            posterior class probabilities are consistently available.
        """
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.random_state = random_state
        self.kwargs = kwargs  # Storage for additional SVC parameters

        # Instantiate the underlying SVC model from scikit-learn
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            probability=True,   # Always enabled for predict_proba()
            random_state=self.random_state,
            **self.kwargs
        )
        
        # Internal state flags
        self.is_fitted = False
        self.classes_ = None

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train):
        """
        Fit the SVM model on the provided training dataset.

        Parameters:
            X_train (array-like): Training feature vectors.
            y_train (array-like): Corresponding class labels.

        Returns:
            self: Fitted classifier instance.
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        self.classes_ = np.unique(y_train)
        return self

    # ------------------------------------------------------------------
    def predict(self, X):
        """
        Predict class labels for the input data.

        Parameters:
            X (array-like): Input feature vectors.

        Returns:
            ndarray: Predicted class labels.
        """
        self._check_fitted()
        return self.model.predict(X)

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """
        Predict posterior probabilities for each class.

        Parameters:
            X (array-like): Input feature vectors.

        Returns:
            ndarray: Probability distribution over classes.
        """
        self._check_fitted()
        return self.model.predict_proba(X)

    # ------------------------------------------------------------------
    def score(self, X, y):
        """
        Compute the classification accuracy on the given dataset.

        Parameters:
            X (array-like): Feature vectors.
            y (array-like): Ground-truth labels.

        Returns:
            float: Accuracy score.
        """
        self._check_fitted()
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    # ------------------------------------------------------------------
    def get_params(self, deep=True):
        """
        Retrieve model hyperparameters in a format compatible with
        scikit-learn utilities such as GridSearchCV.

        Returns:
            dict: Dictionary containing model parameters.
        """
        params = {
            'C': self.C,
            'kernel': self.kernel,
            'degree': self.degree,
            'gamma': self.gamma,
            'random_state': self.random_state,
        }
        params.update(self.kwargs)
        return params

    # ------------------------------------------------------------------
    def set_params(self, **params):
        """
        Update hyperparameters following the scikit-learn parameter
        management protocol.

        Parameters:
            **params: Hyperparameters to update.

        Returns:
            self: Updated classifier instance.
        """
        # Update class attributes
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value

        # Reconstruct parameter dictionary
        all_params = self.get_params()

        # Filter parameters valid for SVC
        svc_params = {
            k: v for k, v in all_params.items()
            if k in SVC()._get_param_names() or k in self.kwargs
        }

        # Update the underlying SVC model
        self.model.set_params(**svc_params)
        return self

    # ------------------------------------------------------------------
    def predict_with_debug(self, X, y_true=None, show_samples=10):
        """
        Return predicted labels and probabilities without printing.
        This method exists to support debugging workflows that require
        access to raw prediction outputs without affecting console output.

        Parameters:
            X (array-like): Input feature vectors.
            y_true (array-like, optional): True labels for comparison.
            show_samples (int): Unused, reserved for future development.

        Returns:
            tuple: (predicted labels, probability estimates)
        """
        self._check_fitted()
        y_proba = self.predict_proba(X)
        y_pred = self.predict(X)
        return y_pred, y_proba

    # ------------------------------------------------------------------
    def _check_fitted(self):
        """
        Verify that the model has been trained before attempting inference.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("The model has not been trained. Call .fit() before inference.")
