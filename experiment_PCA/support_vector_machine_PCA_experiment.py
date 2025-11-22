# SVM_experiment_PCA.py
import os
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize   
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, auc)
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from models.Support_Vector_Machine import UnifiedSVMClassifier


def SVM_experiment_PCA(
    input_path,
    output_dir,
    output_file_name="SVM_PCA_SMOTE_results",
    label_col="Result",
    n_splits=5,
    scalers=None,
    random_state=42
):
    """
    Thí nghiệm SVM sau PCA (2D) + SMOTE
    Hỗ trợ cả Binary và Multiclass classification
    Dùng Stratified K-Fold Cross Validation
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Đọc dữ liệu đã SMOTE
    df = pd.read_csv(input_path)
    print(f"Đã tải dữ liệu: {input_path} | Shape: {df.shape}")

    X = df.drop(columns=[label_col])
    y = df[label_col].copy()

    # Phát hiện multiclass
    classes = sorted(y.unique())
    n_classes = len(classes)
    is_multiclass = n_classes > 2
    print(f"Phát hiện: {'Multiclass' if is_multiclass else 'Binary'} "
          f"({n_classes} lớp: {classes})")

    # Chuẩn hóa nhãn về 0,1,2,...
    if y.min() >= 1:
        y = y - y.min()

    class_names = [f"Stage {c}" if label_col == "Stage" else f"Class {c}" for c in classes]

    # 2. PCA 2 components
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"PCA hoàn tất | Tổng explained variance (2D): {explained_variance:.3%}")

    # Visualize PCA
    pca_vis_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_vis_df[label_col] = y.values
    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=pca_vis_df, x='PC1', y='PC2', hue=label_col, palette='deep', alpha=0.8, s=60)
    plt.title(f'PCA 2D Visualization\nAfter SMOTE | Total Variance: {explained_variance:.1%}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.legend(title=label_col)
    plt.grid(True, alpha=0.3)
    pca_path = os.path.join(output_dir, "PCA_2D_Visualization.png")
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. KFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Scalers
    if scalers is None:
        scalers = {
            'NoScaler': None,
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer': Normalizer()
        }

    # Models
    models = {
        'LinearSVC': UnifiedSVMClassifier(svm_type="linear_svc", C=1.0, calibration=True, random_state=random_state),
        'SGD_Hinge': UnifiedSVMClassifier(svm_type="sgd", loss="hinge", alpha=0.0001, calibration=True, random_state=random_state),
        'SGD_SqHinge': UnifiedSVMClassifier(svm_type="sgd", loss="squared_hinge", alpha=0.0001, calibration=True, random_state=random_state),
    }

    fold_results = []
    fold = 1

    for train_idx, test_idx in skf.split(X_pca, y):
        print(f"\n=== Fold {fold}/{n_splits} ===")
        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for scaler_name, scaler in scalers.items():
            if scaler is not None:
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train.copy()
                X_test_scaled = X_test.copy()

            for model_name, model in models.items():
                print(f"  [{scaler_name} + {model_name}] Training...")

                model_dir = os.path.join(output_dir, model_name)
                os.makedirs(model_dir, exist_ok=True)

                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)

                # Accuracy
                acc = accuracy_score(y_test, y_pred)

                # Metrics
                if is_multiclass:
                    pre = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    try:
                        roc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                    except:
                        roc = np.nan
                else:
                    pre = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    f1  = f1_score(y_test, y_pred, zero_division=0)
                    roc = roc_auc_score(y_test, y_proba[:, 1])

                # Confusion Matrix values
                cm = confusion_matrix(y_test, y_pred)
                tp = tn = fp = fn = None
                if not is_multiclass and cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()

                fold_results.append({
                    'Fold': fold,
                    'Scaler': scaler_name,
                    'Model': model_name,
                    'ACC': round(acc, 4),
                    'Precision': round(pre, 4),
                    'Recall': round(rec, 4),
                    'F1': round(f1, 4),
                    'ROC-AUC': round(roc, 4) if not np.isnan(roc) else 'N/A',
                    'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
                })

                # ROC Curve
                plt.figure(figsize=(8, 6))
                if is_multiclass:
                    y_test_bin = label_binarize(y_test, classes=classes)
                    for i, cls in enumerate(classes):
                        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                        roc_auc_val = auc(fpr, tpr)
                        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc_val:.3f})')
                    plt.title(f'ROC One-vs-Rest - Fold {fold}\n{scaler_name} + {model_name}')
                else:
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                    roc_auc_val = auc(fpr, tpr)
                    plt.plot(fpr, tpr, color='darkorange', lw=2,
                             label=f'ROC (AUC = {roc_auc_val:.3f})')
                    plt.plot([0, 1], [0, 1], 'k--', lw=2)

                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend(loc="lower right")
                roc_path = os.path.join(model_dir, f"ROC_Fold{fold}_{scaler_name}_{model_name}.png")
                plt.savefig(roc_path, dpi=300, bbox_inches='tight')
                plt.close()

                # Confusion Matrix
                plt.figure(figsize=(7, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=class_names, yticklabels=class_names)
                plt.title(f'Confusion Matrix - Fold {fold}\n{scaler_name} + {model_name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                cm_path = os.path.join(model_dir, f"CM_Fold{fold}_{scaler_name}_{model_name}.png")
                plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                plt.close()

        fold += 1

    # Tổng hợp kết quả
    results_df = pd.DataFrame(fold_results)
    summary = results_df.groupby(['Scaler', 'Model']).agg({
        'ACC': 'mean', 'Precision': 'mean', 'Recall': 'mean',
        'F1': 'mean', 'ROC-AUC': lambda x: np.mean([v for v in x if isinstance(v, float)])
    }).round(4).sort_values(by='ACC', ascending=False).reset_index()

    # Lưu kết quả
    results_df.to_csv(os.path.join(output_dir, f"{output_file_name}_detailed.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, f"{output_file_name}_summary.csv"), index=False)

    print(f"\nHOÀN TẤT THÍ NGHIỆM PCA + SVM ({n_splits}-Fold CV)")
    print(f"→ Loại: {'Multiclass' if is_multiclass else 'Binary'} ({n_classes} lớp)")
    print(f"→ Thư mục: {output_dir}")
    print(f"→ Best: {summary.iloc[0]['Model']} + {summary.iloc[0]['Scaler']} | Mean ACC = {summary.iloc[0]['ACC']}")

    return results_df, summary

# SVM_experiment_PCA(
#     input_path="../Data/data_apply_SMOTE/indian_liver_patient_after_SMOTE.csv",
#     output_dir="../experiment_result/indian_liver_patient/SVM/PCA_5Fold_SMOTE",
#     output_file_name="indian_liver_patient_SMOTE_PCA_SVM_results",
#     label_col="Result",
#     n_splits=5,
#     random_state=42
# )
SVM_experiment_PCA(
    input_path="../Data/data_apply_SMOTE/liver_cirrhosis_after_SMOTE.csv",
    output_dir="../experiment_result/liver_cirrhosis/SVM/PCA_5Fold_SMOTE",
    output_file_name="liver_cirrhosis_SMOTE_PCA_SVM_results",
    label_col="Stage",
    n_splits=5,
    random_state=42
)