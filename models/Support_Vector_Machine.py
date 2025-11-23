from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class UnifiedSVMClassifier(BaseEstimator, ClassifierMixin):
    """
    Một class bọc thống nhất cho 2 triển khai SVM tuyến tính:
    - LinearSVC (LibLinear)
    - SGDClassifier với loss='hinge' hoặc 'squared_hinge' (SGD training)
    
    Luôn có predict_proba nhờ CalibratedClassifierCV (Platt scaling hoặc sigmoid).
    """
    
    def __init__(
        self,
        svm_type="linear_svc",   # "linear_svc" hoặc "sgd"
        C=1.0,
        penalty="l2",
        loss="squared_hinge",   # chỉ dùng khi svm_type="sgd"
        alpha=0.0001,            # chỉ dùng khi svm_type="sgd" (regularization strength)
        max_iter=1000,
        tol=1e-4,
        random_state=None,
        class_weight=None,
        calibration=True,        # bật/tắt calibration để có predict_proba chính xác hơn
    ):
        self.svm_type = svm_type.lower()
        self.C = C
        self.penalty = penalty
        self.loss = loss
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.class_weight = class_weight
        self.calibration = calibration
        
        # Sẽ được tạo trong fit()
        self.model = None
        self.classes_ = None

    def _create_base_model(self):
        if self.svm_type == "linear_svc":
            return LinearSVC(
                C=self.C,
                penalty=self.penalty,
                loss='squared_hinge',   # LinearSVC chỉ hỗ trợ squared_hinge
                dual=True,              # giữ mặc định, tự động chọn khi n_samples > n_features
                tol=self.tol,
                max_iter=self.max_iter,
                random_state=self.random_state,
                class_weight=self.class_weight,
            )
        elif self.svm_type == "sgd":
            return SGDClassifier(
                loss=self.loss,         # 'hinge' = SVM cổ điển, 'squared_hinge' = giống LinearSVC
                penalty=self.penalty,
                alpha=self.alpha,       # alpha = 1/(C * n_samples) trong một số trường hợp
                max_iter=self.max_iter,
                tol=self.tol,
                learning_rate='optimal',
                random_state=self.random_state,
                class_weight=self.class_weight,
            )
        else:
            raise ValueError("svm_type phải là 'linear_svc' hoặc 'sgd'")

    def fit(self, X, y):
        base_model = self._create_base_model()
        
        if self.calibration:
            # Cách này HOÀN HẢO tương thích mọi phiên bản sklearn
            self.model = CalibratedClassifierCV(
                estimator=base_model,   # hoạt động cả cũ lẫn mới (từ sklearn 1.4+ bắt buộc dùng estimator)
                method='sigmoid',
                cv=3                     # dùng 3-fold nội bộ để calibrate tốt hơn (khuyến nghị)
            )
            self.model.fit(X, y)
        else:
            self.model = base_model
            self.model.fit(X, y)
        
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        """Dự đoán nhãn lớp"""
        if self.model is None:
            raise RuntimeError("Model chưa được fit!")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Dự đoán xác suất (luôn có nhờ calibration)"""
        if self.model is None:
            raise RuntimeError("Model chưa được fit!")
            
        if not self.calibration and self.svm_type == "linear_svc":
            raise RuntimeError("LinearSVC gốc không có predict_proba. Hãy bật calibration=True")
            
        return self.model.predict_proba(X)

    def decision_function(self, X):
        """Trả về khoảng cách đến siêu phẳng (raw score)"""
        if self.model is None:
            raise RuntimeError("Model chưa được fit!")
        return self.model.decision_function(X)

    def __repr__(self):
        return f"UnifiedSVMClassifier(svm_type={self.svm_type}, C={self.C}, calibration={self.calibration})"