"""
=============================================================================
Random Forest Stage Prediction (ALL-IN-ONE)
- Config class
- Load dataset
- Stratified K-Fold CV with OOF
- Save per-fold models
- Export metrics to CSV (experiment_result/)
- Save OOF predictions
- Plot confusion matrix + ROC curves
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
)

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")


# =============================================================================
# CONFIG (Teammate-style)
# =============================================================================
class Config:
    """Experiment configuration for Random Forest multiclass Stage prediction."""

    # ---- Data ----
    CSV_PATH = r"D:\1_PTTK_UTILS\dataset\normalized_cirrhosis.csv"
    TEST_CSV_PATH = r"D:\1_PTTK_UTILS\dataset\normalized_cirrhosis.csv"
    TARGET_COL = "Stage"  # multiclass (e.g. 1,2,3 or 1..4)

    FEATURE_COLS = [
        "Drug",
        "Sex",
        "Ascites",
        "Hepatomegaly",
        "Spiders",
        "Edema",
        "Bilirubin",
        "Cholesterol",
        "Albumin",
        "Copper",
        "Alk_Phos",
        "SGOT",
        "Tryglicerides",
        "Platelets",
        "Prothrombin",
    ]

    # ---- Experiment ----
    RANDOM_STATE = 42
    N_SPLITS = 6

    # ---- Model ----
    RF_PARAMS = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",  # works fine for multiclass
    }

    # ---- Model Output (per-fold models) ----
    MODEL_PATH = r"D:\1_PTTK_UTILS\models\rf_stage.pkl"
    ENSEMBLE_MODEL_PATTERN = (
        r"D:\1_PTTK_UTILS\models\rf_stage_fold_{fold}.pkl"
    )

    # ---- Experiment Output (metrics, plots, oof) ----
    EXPERIMENT_OUTPUT_DIR = r"../experiment_result/rf"
    METRICS_CSV_NAME = "rf_cv_metrics.csv"
    OOF_CSV_NAME = "rf_oof_predictions.csv"
    CONF_MAT_PNG = "rf_confusion_matrix.png"
    ROC_PNG = "rf_roc_curves.png"


# =============================================================================
# DATA LOADER
# =============================================================================
def load_dataset(csv_path: str = Config.CSV_PATH):
    """
    Load the preprocessed CSV and return X (features) and y (target).
    """
    df = pd.read_csv(csv_path)

    # Basic sanity check
    missing_cols = [c for c in Config.FEATURE_COLS + [Config.TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    X = df[Config.FEATURE_COLS]
    y = df[Config.TARGET_COL].values  # keep as multiclass labels

    return X, y


# =============================================================================
# MODEL BUILDER
# =============================================================================
def build_model() -> RandomForestClassifier:
    """
    Create a RandomForestClassifier with parameters from Config.
    """
    return RandomForestClassifier(**Config.RF_PARAMS)


# =============================================================================
# RANDOM FOREST EXPERIMENT
# =============================================================================
class RFExperiment:
    """
    Random Forest experiment with Stratified K-Fold OOF evaluation.
    """

    def __init__(self):
        self.n_splits = Config.N_SPLITS
        self.random_state = Config.RANDOM_STATE

        # Where to save stuff
        self.model_dir = os.path.dirname(Config.MODEL_PATH)
        os.makedirs(self.model_dir, exist_ok=True)

        self.exp_dir = Config.EXPERIMENT_OUTPUT_DIR
        os.makedirs(self.exp_dir, exist_ok=True)

        # Tracking
        self.fold_results: list[dict] = []
        self.all_y_true: list[int] = []
        self.all_y_pred: list[int] = []
        self.all_y_proba: list[list[float]] = []

        self.classes_: np.ndarray | None = None
        self.train_time: float = 0.0
        self.test_time: float = 0.0

    # -------------------------------------------------------------------------
    def run(self, X: pd.DataFrame, y: np.ndarray):
        """
        Run Stratified K-Fold CV with OOF predictions.
        """
        print("=" * 80)
        print("RANDOM FOREST STAGE PREDICTION - K-FOLD OOF EXPERIMENT")
        print("=" * 80)
        print(f"CSV            : {Config.CSV_PATH}")
        print(f"Features ({len(Config.FEATURE_COLS)}): {Config.FEATURE_COLS}")
        print(f"Target         : {Config.TARGET_COL}")
        print(f"Splits         : {self.n_splits}")
        print(f"Random State   : {self.random_state}")
        print(f"Models dir     : {self.model_dir}")
        print(f"Experiment dir : {self.exp_dir}")
        print("=" * 80)

        self.classes_ = np.unique(y)
        print("Classes:", self.classes_)

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            print(f"\n📊 Fold {fold}/{self.n_splits}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            print(f"   Train samples: {len(X_train)}")
            print(f"   Val   samples: {len(X_val)}")

            # Build model
            model = build_model()

            # Train
            t0 = time.time()
            model.fit(X_train, y_train)
            t1 = time.time()

            # Save model for this fold
            model_path = Config.ENSEMBLE_MODEL_PATTERN.format(fold=fold)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            print(f"   ✔ Saved model for fold {fold}: {model_path}")

            # Predict
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)
            t2 = time.time()

            self.train_time += (t1 - t0)
            self.test_time += (t2 - t1)

            # Store OOF
            self.all_y_true.extend(y_val.tolist())
            self.all_y_pred.extend(y_pred.tolist())
            self.all_y_proba.extend(y_proba.tolist())

            # Metrics
            metrics = self._calculate_metrics(y_val, y_pred, y_proba)
            metrics["fold"] = fold
            metrics["train_samples"] = len(X_train)
            metrics["val_samples"] = len(X_val)
            self.fold_results.append(metrics)

            print(
                f"   Acc: {metrics['accuracy']:.4f} | "
                f"F1_macro: {metrics['f1_macro']:.4f} | "
                f"ROC_AUC_OVR: {metrics['roc_auc_ovr']:.4f}"
            )

        # After all folds
        self._print_summary()
        self._save_metrics_csv()
        self._save_oof_csv()
        self._plot_confusion_matrix()
        self._plot_roc_curves()

        print("\n" + "=" * 80)
        print("✔ EXPERIMENT COMPLETED")
        print("=" * 80)

    # -------------------------------------------------------------------------
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> dict:
        """
        Calculate all metrics for one fold.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "precision_weighted": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall_weighted": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        }

        # Multiclass ROC-AUC (OVR)
        try:
            metrics["roc_auc_ovr"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except Exception:
            metrics["roc_auc_ovr"] = np.nan

        return metrics

    # -------------------------------------------------------------------------
    def _print_summary(self):
        """
        Print summary across folds + global classification report (OOF).
        """
        print("\n" + "=" * 80)
        print("CV SUMMARY (PER-FOLD METRICS)")
        print("=" * 80)

        df = pd.DataFrame(self.fold_results)

        cols_to_show = [
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "precision_macro",
            "precision_weighted",
            "recall_macro",
            "recall_weighted",
            "roc_auc_ovr",
        ]

        print(f"\n{'Metric':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 80)
        for col in cols_to_show:
            if col in df.columns:
                print(
                    f"{col:<20}"
                    f"{df[col].mean():<10.4f}"
                    f"{df[col].std():<10.4f}"
                    f"{df[col].min():<10.4f}"
                    f"{df[col].max():<10.4f}"
                )

        print(f"\nTotal train time: {self.train_time:.2f}s")
        print(f"Total test  time: {self.test_time:.2f}s")

        # OOF classification report
        y_true = np.array(self.all_y_true)
        y_pred = np.array(self.all_y_pred)

        print("\n" + "=" * 80)
        print("OUT-OF-FOLD CLASSIFICATION REPORT (ALL FOLDS MERGED)")
        print("=" * 80)
        print(classification_report(y_true, y_pred, digits=4))

    # -------------------------------------------------------------------------
    def _save_metrics_csv(self):
        """
        Save per-fold metrics + mean + std into experiment_result folder.
        """
        df = pd.DataFrame(self.fold_results)

        # Mean row
        mean_row = {col: df[col].mean() for col in df.columns if col != "fold"}
        mean_row["fold"] = "MEAN"

        # Std row
        std_row = {col: df[col].std() for col in df.columns if col != "fold"}
        std_row["fold"] = "STD"

        df_out = pd.concat(
            [df, pd.DataFrame([mean_row]), pd.DataFrame([std_row])],
            ignore_index=True,
        )

        out_path = os.path.join(self.exp_dir, Config.METRICS_CSV_NAME)
        df_out.to_csv(out_path, index=False)
        print(f"\n✔ Metrics CSV saved → {out_path}")

    # -------------------------------------------------------------------------
    def _save_oof_csv(self):
        """
        Save OOF predictions and probabilities to CSV in experiment_result.
        """
        y_true = np.array(self.all_y_true)
        y_pred = np.array(self.all_y_pred)
        y_proba = np.array(self.all_y_proba)

        df_oof = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

        if self.classes_ is not None:
            for i, cls in enumerate(self.classes_):
                df_oof[f"proba_class_{cls}"] = y_proba[:, i]

        out_path = os.path.join(self.exp_dir, Config.OOF_CSV_NAME)
        df_oof.to_csv(out_path, index=False)
        print(f"✔ OOF predictions CSV saved → {out_path}")

    # -------------------------------------------------------------------------
    def _plot_confusion_matrix(self):
        """
        Plot and save confusion matrix (counts) from OOF predictions.
        """
        y_true = np.array(self.all_y_true)
        y_pred = np.array(self.all_y_pred)

        cm = confusion_matrix(y_true, y_pred, labels=self.classes_)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.classes_,
            yticklabels=self.classes_,
        )
        plt.title("Random Forest - Confusion Matrix", fontweight="bold")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        out_path = os.path.join(self.exp_dir, Config.CONF_MAT_PNG)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✔ Confusion matrix PNG saved → {out_path}")

    # -------------------------------------------------------------------------
    def _plot_roc_curves(self):
        """
        Plot multiclass ROC curves using OVR strategy.
        """
        if self.classes_ is None or len(self.classes_) < 2:
            print("Skipping ROC plot: not enough classes.")
            return

        y_true = np.array(self.all_y_true)
        y_proba = np.array(self.all_y_proba)

        # Binarize labels
        y_bin = label_binarize(y_true, classes=self.classes_)
        n_classes = y_bin.shape[1]

        plt.figure(figsize=(8, 6))

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"Class {self.classes_[i]} (AUC = {roc_auc:.3f})")

        # Random baseline
        plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Random Forest - Multiclass ROC (OVR)", fontweight="bold")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(self.exp_dir, Config.ROC_PNG)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✔ ROC curves PNG saved → {out_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    X, y = load_dataset()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("Unique Stage labels:", sorted(np.unique(y)))

    experiment = RFExperiment()
    experiment.run(X, y)


if __name__ == "__main__":
    main()
