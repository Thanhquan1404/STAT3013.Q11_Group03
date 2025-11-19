# ====================================================================
# UNIVERSAL EXPERIMENT PIPELINE – K-NEAREST NEIGHBORS (Custom KNN)
# Academic / Research-Grade | Fully Automatic | Binary & Multiclass
# ====================================================================
"""
UNIVERSAL KNN EXPERIMENT PIPELINE
=================================
• Tự động nhận diện dataset (Indian Liver vs Cirrhosis)
• Tự động phát hiện nhãn + task (binary/multiclass)
• Universal preprocessing + scaler factory
• SMOTE chỉ dùng cho binary
• GridSearch tự động tìm K tối ưu + weights + metric
• Lưu kết quả vào file CSV tổng hợp (dễ so sánh với SVM, RF, Logistic...)
• Đo thời gian chính xác
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from src.K_nearest_neighbor import KNNClassifier  

# ===========================================================================
# Scaler Factory (chuẩn như các pipeline trước)
# ===========================================================================
def get_scaler(name: str):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, \
        Normalizer, MaxAbsScaler, QuantileTransformer
    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
        "Normalizer": Normalizer(),
        "MaxAbsScaler": MaxAbsScaler(),
        "QuantileTransformer": QuantileTransformer(output_distribution="normal", random_state=42),
        None: None
    }
    if name not in scalers:
        raise ValueError(f"Unsupported scaler: {name}")
    return scalers[name]

# ===========================================================================
# UNIVERSAL PREPROCESSING (tự động detect tất cả)
# ===========================================================================
def universal_preprocessing(
    path: str,
    scaler_name="StandardScaler",
    apply_smote=True,
    random_state=42
):
    print(f"\n[INFO] Loading dataset: {os.path.basename(path)}")
    df = pd.read_csv(path)

    # Auto-detect label column
    possible_labels = ["Result", "Dataset", "Class", "selector", "target", "Stage", "status", "Diagnosis"]
    label_col = next((c for c in possible_labels if c in df.columns), None)
    if label_col is None:
        raise ValueError("Không tìm thấy cột nhãn!")

    y_raw = df[label_col].copy()

    # Task detection
    unique_vals = np.unique(y_raw.dropna())
    if len(unique_vals) > 2 or label_col == "Stage":
        task = "multiclass"
        y = y_raw.astype(int).values
        n_classes = len(unique_vals)
    else:
        task = "binary"
        n_classes = 2
        pos_label = y_raw.max()
        y = (y_raw == pos_label).astype(int).values

    # Drop missing label
    if y_raw.isnull().any():
        df = df.dropna(subset=[label_col]).reset_index(drop=True)

    X = df.drop(columns=[label_col])

    # Encode categorical
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Impute numerical
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        X[num_cols] = SimpleImputer(strategy="mean").fit_transform(X[num_cols])

    # Scaling (rất quan trọng với KNN!)
    scaler = get_scaler(scaler_name)
    if scaler is not None:
        X = scaler.fit_transform(X)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)

    # Stratified split
    stratify = y if task == "binary" or n_classes > 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=stratify
    )

    # SMOTE chỉ cho binary
    if apply_smote and task == "binary":
        X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)
        print(f"[INFO] SMOTE applied → X_train: {X_train.shape}")

    print(f"[INFO] Task: {task.upper()} | Classes: {n_classes} | Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, y_train, X_test, y_test, task, label_col

# ===========================================================================
# MAIN EXPERIMENT LOOP – KNN
# ===========================================================================
if __name__ == "__main__":

    # ====================================================================
    # CẤU HÌNH CHẠY – CHỈ CẦN THAY 1 DÒNG
    # ====================================================================
    DATASET_PATH = "../data/processed/icirrhosis_encoded.csv"        # Multiclass
    # DATASET_PATH = "../data/processed/liver_cleaned.csv"           # Binary

    OUTPUT_CSV = "../experiment_result/k_nearest_neighbors_cirhosis_result.csv"

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Header CSV (chỉ ghi lần đầu)
    if not os.path.exists(OUTPUT_CSV):
        header = (
            "Dataset,Task,K,Weights,Metric,P,Best_CV_Acc,Test_Accuracy,Test_F1_Weighted,Test_F1_Macro,"
            "Train_Time_s,Test_Time_s,Scaler,Timestamp\n"
        )
        with open(OUTPUT_CSV, "w") as f:
            f.write(header)

    # Các cấu hình muốn thử
    SCALERS = ["StandardScaler", "RobustScaler", "MinMaxScaler"]
    PARAM_GRIDS = [
        {"n_neighbors": 3, "weights": "distance", "metric": "euclidean", "p": 2},
        {"n_neighbors": 5, "weights": "distance", "metric": "euclidean", "p": 2},
        {"n_neighbors": 7, "weights": "distance", "metric": "manhattan", "p": 1},
        {"n_neighbors": 9, "weights": "uniform",  "metric": "euclidean", "p": 2},
    ]

    for scaler_name in SCALERS:
        X_train, y_train, X_test, y_test, task, label_col = universal_preprocessing(
            path=DATASET_PATH,
            scaler_name=scaler_name,
            apply_smote=True,
            random_state=42
        )

        for params in PARAM_GRIDS:
            K = params["n_neighbors"]
            print(f"\n{'='*70}")
            print(f"RUNNING KNN → K={K} | {params['weights']} | {params['metric']} | p={params['p']} | Scaler: {scaler_name}")
            print(f"{'='*70}")

            # Tạo model
            model = KNNClassifier(
                n_neighbors=params["n_neighbors"],
                weights=params["weights"],
                metric=params["metric"],
                p=params["p"]
            )

            # Cross-validation để lấy best CV score
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                model.fit(X_tr, y_tr)
                pred = model.predict(X_val)
                cv_scores.append(accuracy_score(y_val, pred))
            best_cv_acc = np.mean(cv_scores)

            # Train trên toàn bộ train set
            start_train = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_train

            # Predict
            start_test = time.time()
            y_pred = model.predict(X_test)
            test_time = time.time() - start_test

            # Metrics
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            test_acc = report["accuracy"]
            test_f1w = report["weighted avg"]["f1-score"]
            test_f1m = report.get("macro avg", {}).get("f1-score", f1_score(y_test, y_pred, average="macro"))

            # Lưu kết quả
            row = [
                os.path.basename(DATASET_PATH),
                task,
                K,
                params["weights"],
                params["metric"],
                params["p"],
                round(best_cv_acc, 4),
                round(test_acc, 4),
                round(test_f1w, 4),
                round(test_f1m, 4),
                round(train_time, 4),
                round(test_time, 4),
                scaler_name,
                time.strftime("%Y%m%d_%H%M%S")
            ]

            pd.DataFrame([row]).to_csv(OUTPUT_CSV, mode="a", header=False, index=False)

            print(f"SAVED | CV Acc: {best_cv_acc:.4f} → Test Acc: {test_acc:.4f} | F1w: {test_f1w:.4f}")
            print(f"Time → Train: {train_time:.3f}s | Test: {test_time:.4f}s\n")

    print(f"\nHOÀN TẤT! Tất cả kết quả KNN đã được lưu tại:")
    print(f"   → {OUTPUT_CSV}")
    print("   Bây giờ bạn có bảng so sánh đầy đủ: Logistic → SVM → RF → KNN!")