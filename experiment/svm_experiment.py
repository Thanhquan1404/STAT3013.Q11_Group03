# ====================================================================
# File này dùng để GỌI class SVMClassifier và chạy Pipeline
# PHIÊN BẢN: TÁI CẤU TRÚC THEO CHUẨN ML (Đã sửa lỗi In lặp lại)
# ====================================================================

# 1. Import các thư viện cần thiết
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import multiprocessing # <-- THÊM THƯ VIỆN NÀY

try:
    from svmClass import SVMClassifier # <-- Đảm bảo tên file là svmClass.py
except ImportError:
    print("LỖI: Không tìm thấy file 'svmClass.py'.") 
    print("Hãy đảm bảo hai file (svmClass.py và run_svm_example.py) nằm chung thư mục.")
    exit()

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("LỖI: Không tìm thấy thư viện 'imbalanced-learn'.")
    print("Vui lòng chạy lệnh: pip install imbalanced-learn pandas")
    exit()

# ====================================================================
# 1. KHAI BÁO ĐƯỜNG DẪN VÀ CẤU HÌNH (HEADER)
# ====================================================================
DATASET_FILEPATH = "liver_cirrhosis.csv"
TARGET_COLUMN = 'Stage'

# Định nghĩa các cột
CATEGORICAL_FEATURES = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
NUMERICAL_FEATURES = ['Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']
COLUMNS_TO_DROP = ['Status', 'N_Days']
RANDOM_STATE = 42
CV_SPLITS = 5 # K-Fold = 5

# ====================================================================
# 2. HÀM TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU (PREPROCESSING)
# ====================================================================
def load_and_preprocess_data(filepath):
    """Tải dữ liệu từ CSV, làm sạch cơ bản và tách X, y."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return None, None, None, None # Thêm None thứ 4 cho display_labels

    # Loại bỏ các cột không dùng
    df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')
    
    # Loại bỏ các hàng có target bị thiếu
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    # Tự động lấy nhãn hiển thị
    display_labels = [str(label) for label in sorted(y.unique())]
    
    return X, y, display_labels, df # Trả về df để lấy thông tin tổng kết

# ====================================================================
# 3. HÀM KHỞI TẠO SCALER VÀ PIPELINE
# ====================================================================
def create_pipeline(numerical_features, categorical_features):
    """Tạo ra pipeline hoàn chỉnh (bao gồm Imputer, Scaler, SMOTE, Model)."""
    
    # 4.1: Pipeline con cho dữ liệu SỐ
    numeric_transformer = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # Xử lý NaN
        ('scaler', StandardScaler())                 # Scale
    ])

    # 4.2: Pipeline con cho dữ liệu PHÂN LOẠI
    categorical_transformer = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Xử lý NaN
        ('onehot', OneHotEncoder(handle_unknown='ignore'))    # Chuyển thành số
    ])

    # 4.3: Gộp 2 pipeline con bằng ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' 
    )
    
    # 4.4: Tạo Pipeline tổng
    svm_model_template = SVMClassifier(random_state=RANDOM_STATE)
    
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),              # Bước 1: Tiền xử lý (Impute, Scale, OneHot)
        ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=3)), # Bước 2: SMOTE
        ('model', svm_model_template)                 # Bước 3: Chạy model
    ])
    
    return pipeline

# ====================================================================
# 4. HÀM ĐÁNH GIÁ (EVALUATE)
# ====================================================================
def evaluate_model(y_true, y_pred, display_labels, model_name="Model"):
    """In báo cáo classification report."""
    print("\n" + "="*80)
    print(f"--- Báo cáo chi tiết trên tập TEST (dùng {model_name}) ---")
    print(f"--- (Các nhãn {sorted(y_true.unique())} sẽ được hiển thị là {display_labels}) ---")
    
    print(classification_report(y_true, y_pred, target_names=display_labels, zero_division=0))
    print("="*80)

def print_grid_search_summary(grid_search, cv_splits):
    """In kết quả chi tiết và tóm tắt từ GridSearchCV."""
    
    cv_results = pd.DataFrame(grid_search.cv_results_)
    
    # 8. BỔ SUNG: In kết quả của TẤT CẢ các model đã thử
    print("\n" + "="*80)
    print(f"--- Báo cáo chi tiết TẤT CẢ KẾT QUẢ TỪNG MODEL (CV={cv_splits}) ---")
    
    cols_to_show = [
        'param_model__kernel', 'param_model__C', 'param_model__degree', 
        'param_model__gamma', 'mean_test_score', 'std_test_score', 'rank_test_score'
    ]
    existing_cols = [col for col in cols_to_show if col in cv_results.columns]
    results_df = cv_results[existing_cols].sort_values(by='rank_test_score')
    
    print(results_df.to_string(index=False)) 
    print("="*80)

    # 9. BỔ SUNG: TÓM TẮT ĐIỂM F1 TỐT NHẤT CHO TỪNG KERNEL
    print("\n" + "="*80)
    print(f"--- TÓM TẮT: ĐIỂM F1-SCORE TỐT NHẤT CHO TỪNG KERNEL (CV={cv_splits}) ---")
    
    kernel_scores = cv_results[['param_model__kernel', 'mean_test_score']]
    summary_df = kernel_scores.groupby('param_model__kernel').max()
    summary_df = summary_df.sort_values(by='mean_test_score', ascending=False)
    
    print(summary_df.rename(columns={'mean_test_score': 'Best F1-Score'}))
    print("="*80)

# ====================================================================
# 5. HÀM KHỞI CHẠY CHÍNH (MAIN)
# ====================================================================
def main_runner():
    """Hàm chính để chạy toàn bộ quy trình."""
    
    # 1. Tải và xử lý dữ liệu
    X, y, display_labels, df = load_and_preprocess_data(DATASET_FILEPATH)
    
    if X is None:
        print(f"LỖI: Không tìm thấy file '{DATASET_FILEPATH}'.")
        print("Hãy đảm bảo file .csv nằm chung thư mục với file run_svm_example.py")
        return # Thoát nếu không tải được file

    # --- IN THÔNG TIN TỔNG KẾT SAU KHI ĐÃ TẢI THÀNH CÔNG ---
    print("="*60)
    print(" BẮT ĐẦU VÍ DỤ CHẠY SVM (CẤU TRÚC CHUẨN)")
    print("="*60)
    print(f"Tải dữ liệu thành công. Tổng số mẫu: {len(df)}")
    print(f"Phát hiện các nhãn (Stages): {display_labels}")
    print(f"Phân bố lớp (trước SMOTE):\n{y.value_counts().sort_index()}")
    print(f"Tổng số giá trị bị khuyết (NaN) trong X: {X.isnull().sum().sum()}")
    
    # 2. Chia dữ liệu (Data Split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Kích thước Train: {X_train.shape}, Kích thước Test: {X_test.shape}")

    # 3. Tạo Pipeline
    pipeline = create_pipeline(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)

    # 4. Hiệu chỉnh mô hình (Grid Search)
    
    # --- LƯỚI THỬ NGHIỆM TÙY CHỈNH ---
    print("\nĐANG CHẠY")
    param_grid = [
        {'model__kernel': ['linear'], 'model__C': [0.1,1,10,100]},
        {'model__kernel': ['rbf'], 'model__C': [97,98,99], 'model__gamma': ['scale',0.4, 0.5,0.6]},
        {'model__kernel': ['poly'], 'model__C': [1,10,100], 'model__degree': [2], 'model__gamma': ['scale',2,3]}
    ]
    
    cv_strategy = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print("\n" + "="*60)
    print(f"Bắt đầu GridSearchCV (với KFold={CV_SPLITS}) trên 'liver_cirrhosis'...")
    print("Vui lòng đợi...")
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid, 
        cv=cv_strategy, 
        scoring='f1_weighted', 
        n_jobs=-1, 
        verbose=1  
    )
    
    grid_search.fit(X_train, y_train)
    
    # 5. Xem kết quả
    print("\n" + "="*60)
    print("GRID SEARCH HOÀN TẤT!")
    print(f"Tham số tốt nhất (Best Params): \n{grid_search.best_params_}")
    print(f"\nBest F1-score (trên tập Train CV): {grid_search.best_score_:.4f}")
    
    best_pipeline = grid_search.best_estimator_

    # 6. Đánh giá cuối cùng trên tập TEST
    y_pred = best_pipeline.predict(X_test)
    evaluate_model(y_test, y_pred, display_labels, model_name="Model TỐT NHẤT")
    
    # 7. In bảng tóm tắt
    print_grid_search_summary(grid_search, CV_SPLITS)

    print("\n" + "="*60)
    print(" VÍ DỤ VỚI CẤU TRÚC CHUẨN ĐÃ CHẠY XONG!")
    print("="*60)

# Dòng này đảm bảo code chỉ chạy khi bạn thực thi file này trực tiếp
# CẬP NHẬT: Thêm kiểm tra đa luồng (multiprocessing)
if __name__ == "__main__":
    # Chỉ chạy main_runner nếu đây là tiến trình chính
    if multiprocessing.current_process().name == 'MainProcess':
        main_runner()
    # Nếu không phải tiến trình chính, GridSearch sẽ chạy tiếp mà không in thông báo
