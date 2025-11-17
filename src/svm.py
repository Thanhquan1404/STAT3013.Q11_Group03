import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline # Vô hiệu hóa pipeline gốc của sklearn
from imblearn.pipeline import Pipeline as ImbPipeline # THÊM: Sử dụng pipeline của imblearn
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import chi2_contingency, spearmanr
import warnings

# THÊM: Thư viện cho SMOTE
from imblearn.over_sampling import SMOTE

# Bỏ qua các cảnh báo
warnings.filterwarnings("ignore")

# ====================================================================
# PHẦN 1: TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
# ====================================================================

print("="*60)
print("PHẦN 1: TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU")
print("="*60)

# Tải dữ liệu
try:
    df = pd.read_csv("liver_cirrhosis.csv")
except FileNotFoundError:
    print("LỖI: Không tìm thấy file 'liver_cirrhosis.csv'.")
    exit()

# Xử lý dữ liệu thiếu (NA) và chuyển đổi kiểu dữ liệu
df.dropna(inplace=True)
df['Stage'] = df['Stage'].astype(int)

# Định nghĩa các đặc trưng và biến mục tiêu
categorical_features_all = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
numerical_features_all = ['Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']
target = 'Stage'

# Loại bỏ các cột không phải là đặc trưng dự đoán
columns_to_drop = ['Status', 'N_Days']
df = df.drop(columns=columns_to_drop, errors='ignore')

# Cập nhật lại danh sách numerical_features
numerical_features_all = [col for col in numerical_features_all if col not in columns_to_drop]

print(f"Số mẫu sau khi loại bỏ NA: {len(df)}")
print(f"Phân bố các Giai đoạn (Stage):\n{df['Stage'].value_counts().sort_index()}")
print("\n" + "="*60)

# ====================================================================
# PHẦN 2: PHÂN TÍCH THỐNG KÊ & LỰA CHỌN ĐẶC TRƯNG
# ====================================================================
print("PHẦN 2: PHÂN TÍCH THỐNG KÊ (LỰA CHỌN ĐẶC TRƯNG)")
print("="*60)

features_to_drop = []
p_value_threshold = 0.05 # Ngưỡng ý nghĩa thống kê

# --- 2.1. Kiểm tra Chi-square (Biến phân loại vs Stage) ---
print("--- 2.1. Kết quả Kiểm tra Chi-square (Categorical vs Target) ---")
for col in categorical_features_all:
    contingency_table = pd.crosstab(df[col], df[target])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    is_significant = "Có ý nghĩa" if p < p_value_threshold else "KHÔNG ý nghĩa"
    print(f"Feature: {col:<15} Chi2: {chi2:10.2f}, p-value: {p:.5f} ({is_significant})")

    if p >= p_value_threshold:
        features_to_drop.append(col)

# --- 2.2. Spearman Correlation (Biến liên tục vs Stage) ---
print("\n--- 2.2. Kết quả Hệ số Tương quan Spearman (Numerical vs Target) ---")
for col in numerical_features_all:
    corr, p = spearmanr(df[col], df[target])
    is_significant = "Có ý nghĩa" if p < p_value_threshold else "KHÔNG ý nghĩa"
    print(f"Feature: {col:<15} Correlation (rho): {corr:8.4f}, p-value: {p:.5f} ({is_significant})")

    if p >= p_value_threshold:
        features_to_drop.append(col)

# --- 2.3. Tổng kết các đặc trưng được lựa chọn ---
print("\n--- 2.3. Kết quả Lựa chọn Đặc trưng ---")
print(f"Các đặc trưng có p-value >= 0.05 (sẽ bị loại bỏ): {features_to_drop}")

# Tạo danh sách đặc trưng mới (đã được lựa chọn)
categorical_features_selected = [f for f in categorical_features_all if f not in features_to_drop]
numerical_features_selected = [f for f in numerical_features_all if f not in features_to_drop]

print(f"Đặc trưng Phân loại còn lại: {categorical_features_selected}")
print(f"Đặc trưng Liên tục còn lại: {numerical_features_selected}")
print("\n" + "="*60)
# ====================================================================
# PHẦN 3: THIẾT KẾ PIPELINE VÀ HÀM THỰC NGHIỆM
# ====================================================================
print("PHẦN 3: THIẾT KẾ PIPELINE THỰC NGHIỆM (TÁI SỬ DỤNG)")
print("="*60)

# CẬP NHẬT: Thêm 'kernel_to_test' để chỉ định chạy kernel nào
def run_svm_experiment(X_data, y_data, numerical_features, categorical_features, experiment_name, kernel_to_test):
    """
    Hàm này đóng gói toàn bộ quy trình thực nghiệm SVM cho MỘT kernel CỤ THỂ.

    Nó thực hiện:
    1. Phân chia Train/Test (Stratified)
    2. Định nghĩa Pipeline (Preprocessor + SMOTE + Model)
    3. Định nghĩa Lưới Tham số (Grid Search) CHỈ cho kernel được yêu cầu
    4. Huấn luyện và Tinh chỉnh (với KFold=5)
    5. Đánh giá và In kết quả
    """
    print(f"\n--- BẮT ĐẦU THÍ NGHIỆM: {experiment_name} ---")
    print(f"--- Kernel đang kiểm tra: {kernel_to_test.upper()} ---")

    # 1. Phân chia dữ liệu (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data,
        test_size=0.3,
        random_state=14,
        stratify=y_data
    )
    print(f"Kích thước tập huấn luyện: {X_train.shape}")
    print(f"Kích thước tập kiểm tra: {X_test.shape}")

    # 2. Định nghĩa Pipeline

    # Bước 2.1: Preprocessor (Tiền xử lý)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Bước 2.2: Pipeline hoàn chỉnh (với SMOTE)
    # SMOTE: Xử lý mất cân bằng
    # SVC: Mô hình SVM
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('svc', SVC(random_state=42, kernel=kernel_to_test)) # Gán kernel cố định ở đây
    ])

    # 3. Định nghĩa Lưới Tham số (Hyperparameter Grid)
    # CẬP NHẬT: Tạo grid CỤ THỂ cho kernel được yêu cầu

    param_grid = {} # Khởi tạo grid rỗng

    if kernel_to_test == 'linear':
        param_grid = {
            'svc__C': [0.1, 1, 10]
            # Kernel đã được gán 'linear' trong pipeline
        }

    elif kernel_to_test == 'rbf':
        param_grid = {
            'svc__C': [99],
            'svc__gamma': [0.5]
            # Kernel đã được gán 'rbf' trong pipeline
        }

    elif kernel_to_test == 'poly':
        param_grid = {
            'svc__C': [0.1, 1, 10],
            'svc__degree': [2, 3] # Thử nghiệm đa thức bậc 2 và 3
            # Kernel đã được gán 'poly' trong pipeline
        }
    else:
        raise ValueError(f"Kernel '{kernel_to_test}' không được hỗ trợ.")

    # 4. Huấn luyện và Tinh chỉnh (Grid Search)
    # Yêu cầu: KFold=5
    # StratifiedKFold: Chia K-Fold (n_splits=5) nhưng vẫn giữ tỷ lệ lớp.
    cv_method = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        pipeline,
        param_grid, # CẬP NHẬT: Sử dụng grid của kernel cụ thể
        cv=cv_method,
        scoring='f1_weighted',
        verbose=2,
        n_jobs=-1
    )

    print(f"\nBắt đầu Grid Search (Tinh chỉnh tham số cho kernel {kernel_to_test})...")
    grid_search.fit(X_train, y_train)

    # 5. Đánh giá và In kết quả
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    print("\n--- Kết quả Tinh chỉnh Tốt nhất ---")
    print(f"Tham số Tốt nhất: {best_params}")
    print(f"F1-Score (Weighted) Tốt nhất trên tập Huấn luyện (CV=5): {best_score:.4f}")

    # Đánh giá trên tập Test
    y_pred = best_model.predict(X_test)

    print("\n--- Báo cáo Đánh giá trên tập TEST ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("--- Ma trận Nhầm lẫn (Confusion Matrix) trên tập TEST ---")
    print(confusion_matrix(y_test, y_pred))

    print(f"--- KẾT THÚC THÍ NGHIỆM: {experiment_name} ---\n")
    return best_model

# ====================================================================
# PHẦN 4: CHẠY CÁC THÍ NGHIỆM
# ====================================================================
print("PHẦN 4: CHẠY CÁC THÍ NGHIỆM SO SÁNH (6 THÍ NGHIỆM RIÊNG BIỆT)")
print("="*60)

# Định nghĩa dữ liệu đầu vào chung
X = df.drop(columns=[target])
y = df[target]

"""# --- Nhóm Thí nghiệm 1: Sử dụng TẤT CẢ các đặc trưng ---
print(">>> CHUẨN BỊ NHÓM 1: SỬ DỤNG TẤT CẢ ĐẶC TRƯNG <<<")
X_all = X[categorical_features_all + numerical_features_all]

run_svm_experiment(
    X_all, y,
    numerical_features_all, categorical_features_all,
    "Mô hình 1.1: Tất cả Features + Linear SVM",
    kernel_to_test='linear'
)
run_svm_experiment(
    X_all, y,
    numerical_features_all, categorical_features_all,
    "Mô hình 1.2: Tất cả Features + RBF SVM",
    kernel_to_test='rbf'
)
run_svm_experiment(
    X_all, y,
    numerical_features_all, categorical_features_all,
    "Mô hình 1.3: Tất cả Features + Poly SVM",
    kernel_to_test='poly'
)"""

# --- Nhóm Thí nghiệm 2: Chỉ sử dụng các đặc trưng CÓ Ý NGHĨA ---
print(">>> CHUẨN BỊ NHÓM 2: SỬ DỤNG ĐẶC TRƯNG CHỌN LỌC <<<")
X_selected = X[categorical_features_selected + numerical_features_selected]

run_svm_experiment(
    X_selected, y,
    numerical_features_selected, categorical_features_selected,
    "Mô hình 2.1: Features Chọn lọc + Linear SVM",
    kernel_to_test='linear'
)
run_svm_experiment(
    X_selected, y,
    numerical_features_selected, categorical_features_selected,
    "Mô hình 2.2: Features Chọn lọc + RBF SVM",
    kernel_to_test='rbf'
)
run_svm_experiment(
    X_selected, y,
    numerical_features_selected, categorical_features_selected,
    "Mô hình 2.3: Features Chọn lọc + Poly SVM",
    kernel_to_test='poly'
)


print("\n" + "="*60)
print("TẤT CẢ 6 THÍ NGHIỆM RIÊNG BIỆT ĐÃ HOÀN TẤT.")

print("="*60)
