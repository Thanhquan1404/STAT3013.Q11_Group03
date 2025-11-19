# ====================================================================
# UNIVERSAL EXPERIMENT PIPELINE – XGBoost (Custom Implementation)
# Academic / Research-Grade | Fully Automatic | Binary & Multiclass
# ====================================================================
"""
UNIVERSAL XGBoost EXPERIMENT PIPELINE
=====================================
• Tự động nhận diện dataset (Indian Liver vs Cirrhosis)
• Tự động phát hiện nhãn + task (binary/multiclass)
• Universal preprocessing + scaler factory
• SMOTE chỉ dùng cho binary
• Thử nhiều cấu hình eta, depth, subsample...
• Lưu kết quả vào file CSV tổng hợp (dễ so sánh với SVM, RF, KNN...)
• Đo thời gian train/test chính xác
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from src.XGBoost import XGBoostClassifier  # ← Custom XGBoost của bạn

# ===========================================================================
# Scaler Factory (chuẩn hệ thống)
# ===========================================================================
def get_scaler(name: str):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, MaxAbsScaler, QuantileTransformer
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

    # Task detection + chuẩn hóa nhãn cho XGBoost
    unique_vals = sorted(y_raw.dropna().unique().astype(int))
    if len(unique_vals) > 2 or label_col == "Stage":
        task = "multiclass"
        n_classes = len(unique_vals)
        # Remap nhãn về 0, 1, 2, ..., n-1 (XGBoost bắt buộc!)
        label_mapping = {old: new for new, old in enumerate(unique_vals)}
        y = y_raw.map(label_mapping).astype(int).values
        print(f"[INFO] Multiclass detected → Remapped labels: {unique_vals} → [0, {n_classes-1}]")
    else:
        task = "binary"
        n_classes = 2
        pos_label = y_raw.max()
        y = (y_raw == pos_label).astype(int).values  # 0 và 1
        print(f"[INFO] Binary task → Positive class: {int(pos_label)}")

    # Drop missing label
    if y_raw.isnull().any():
        df = df.dropna(subset=[label_col]).reset_index(drop=True)

    X = df.drop(columns=[label_col])

    # Lưu tên cột trước khi scale (để feature importance)
    feature_names = X.columns.tolist()

    # Encode categorical
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Impute numerical
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        X[num_cols] = SimpleImputer(strategy="mean").fit_transform(X[num_cols])

    # Scaling
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
    
    return X_train, y_train, X_test, y_test, task, label_col, feature_names, unique_vals  


# ===========================================================================
if __name__ == "__main__":
    # DATASET_PATH = "../data/processed/indian_liver_patient_preprocessed.csv"
    # OUTPUT_CSV = "../experiment_result/xgboost_indian_liver_patient_result.csv"

    DATASET_PATH = "../data/processed/liver_cirrhosis_preprocessed.csv"
    OUTPUT_CSV = "../experiment_result/xgboost_liver_cirrhosis_result.csv"
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    if not os.path.exists(OUTPUT_CSV):
        header = "Dataset,Task,Eta,Max_Depth,Subsample,Colsample,Num_Boost,Best_Iter,Test_Accuracy,Test_F1_Weighted,Test_F1_Macro,Train_Time_s,Test_Time_s,Scaler,Timestamp\n"
        with open(OUTPUT_CSV, "w") as f:
            f.write(header)

    XGB_CONFIGS = [
        {"eta": 0.05, "max_depth": 6,  "subsample": 0.8, "colsample_bytree": 0.8, "num_boost_round": 1000},
        {"eta": 0.1,  "max_depth": 8,  "subsample": 0.9, "colsample_bytree": 0.9, "num_boost_round": 800},
        {"eta": 0.03, "max_depth": 10, "subsample": 1.0, "colsample_bytree": 1.0, "num_boost_round": 1500},
    ]

    SCALERS = [None, "StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer","MaxAbsScaler","QuantileTransformer"]

    for scaler_name in SCALERS:
        X_train, y_train, X_test, y_test, task, label_col, feature_names, original_labels = universal_preprocessing(
            path=DATASET_PATH,
            scaler_name=scaler_name,
            apply_smote=True,
            random_state=42
        )

        for cfg in XGB_CONFIGS:
            print(f"\n{'='*80}")
            print(f"XGBOOST → η={cfg['eta']} | depth={cfg['max_depth']} | scaler={scaler_name}")
            print(f"{'='*80}")

            model = XGBoostClassifier(
                eta=cfg["eta"],
                max_depth=cfg["max_depth"],
                subsample=cfg["subsample"],
                colsample_bytree=cfg["colsample_bytree"],
                num_boost_round=cfg["num_boost_round"],
                early_stopping_rounds=50,
                verbose=False
            )

            start_train = time.time()
            model.fit(X_train, y_train, X_test, y_test)
            train_time = time.time() - start_train

            start_test = time.time()
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            test_time = time.time() - start_test

            # Chuyển lại nhãn gốc để tính metrics đúng
            if task == "multiclass":
                inv_map = {i: label for i, label in enumerate(original_labels)}
                y_test_orig = np.array([inv_map[label] for label in y_test])
                y_pred_orig = np.array([inv_map[label] for label in y_pred])
            else:
                y_test_orig = y_test
                y_pred_orig = y_pred

            report = classification_report(y_test_orig, y_pred_orig, output_dict=True, zero_division=0)
            acc = report["accuracy"]
            f1w = report["weighted avg"]["f1-score"]
            f1m = report.get("macro avg", {}).get("f1-score", f1_score(y_test_orig, y_pred_orig, average="macro"))

            best_iter = model.model.best_iteration if hasattr(model.model, 'best_iteration') else cfg["num_boost_round"]

            row = [
                os.path.basename(DATASET_PATH), task, cfg["eta"], cfg["max_depth"],
                cfg["subsample"], cfg["colsample_bytree"], cfg["num_boost_round"], best_iter,
                round(acc, 4), round(f1w, 4), round(f1m, 4),
                round(train_time, 4), round(test_time, 4), scaler_name,
                time.strftime("%Y%m%d_%H%M%S")
            ]
            pd.DataFrame([row]).to_csv(OUTPUT_CSV, mode="a", header=False, index=False)

            print(f"SAVED | Acc: {acc:.4f} | F1w: {f1w:.4f} | Best Iter: {best_iter}\n")

    print(f"\nHOÀN TẤT! Kết quả XGBoost đã lưu tại: {OUTPUT_CSV}")