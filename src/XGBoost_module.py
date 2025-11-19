import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score


class XGBoostClassifier:
    """
    XGBoost Classifier wrapper with sklearn-style interface.
    Supports both binary and multi-class classification.
    """

    def __init__(self, objective="multi:softmax", eta=0.05, max_depth=6,
                 subsample=0.8, colsample_bytree=0.8, num_boost_round=500,
                 early_stopping_rounds=None, verbose=True):

        self.objective = objective
        self.eta = eta
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

        self.model = None
        self.num_class = None
        self.evals_result = {}

    # ======================================================================
    # FIT
    # ======================================================================
    def fit(self, X_train, y_train, X_val=None, y_val=None):

        unique_classes = np.unique(y_train)
        self.num_class = len(unique_classes)

        dtrain = xgb.DMatrix(X_train, label=y_train)

        # AUTO OBJECTIVE
        if self.num_class == 2:
            obj = "binary:logistic"        
        else:
            obj = "multi:softprob"         

        params = {
            "objective": obj,
            "eta": self.eta,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "eval_metric": "logloss" if self.num_class == 2 else "mlogloss"
        }

        if self.num_class > 2:
            params["num_class"] = self.num_class

        evals = [(dtrain, "train")]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "validation"))

        if self.verbose:
            print(f"Training started | eta={self.eta}, depth={self.max_depth}, rounds={self.num_boost_round}")
            print(f"Classes={self.num_class} | Objective={params['objective']}")

        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            evals_result=self.evals_result,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=100 if self.verbose else False
        )

        if self.verbose:
            final_round = self.model.best_iteration if self.early_stopping_rounds else self.num_boost_round
            print(f"Training completed after {final_round} rounds.")

        return self

    # ======================================================================
    # PREDICT PROBA
    # ======================================================================
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet.")

        dtest = xgb.DMatrix(X)

    # BINARY
        if self.num_class == 2:
            p1 = self.model.predict(dtest)
            p0 = 1 - p1
            return np.vstack([p0, p1]).T

    # MULTICLASS 
        return self.model.predict(dtest)  


    # ======================================================================
    # PREDICT 
    # ======================================================================
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained.")

        dtest = xgb.DMatrix(X)

        # ----------------------------
        # BINARY
        # ----------------------------
        if self.num_class == 2:
            prob = self.model.predict(dtest)     
            return (prob >= 0.5).astype(int)

        # ----------------------------
        # MULTI
        # ----------------------------
        proba = self.model.predict(dtest)   
        return np.argmax(proba, axis=1)

    # ======================================================================
    # SCORE
    # ======================================================================
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    # ======================================================================
    # DEBUG PRINT
    # ======================================================================
    def predict_with_debug(self, X, y_true=None, show_samples=10):

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        print("\n" + "=" * 80)
        print("                 PREDICTION DETAILS")
        print("=" * 80)

        header = f"{'Idx':<4} {'True':<6} {'Pred':<6}"
        for c in range(y_proba.shape[1]):
            header += f"P({c})      "
        header += "Correct?"
        print(header)
        print("-" * 80)

        for i in range(min(show_samples, len(y_pred))):

            true = y_true[i] if y_true is not None else "-"
            correct = "✔️" if y_true is not None and y_pred[i] == y_true[i] else "✖️"

            row = f"{i:<4} {true:<6} {y_pred[i]:<6}"
            for prob in y_proba[i]:
                row += f"{prob:<10.4f}"
            row += correct
            print(row)

        if y_true is not None:
            acc = accuracy_score(y_true, y_pred)
            print("-" * 80)
            print(f"Accuracy: {acc:.4f}")
            print("\n" + classification_report(y_true, y_pred))

        print("=" * 80 + "\n")
        return y_pred, y_proba
