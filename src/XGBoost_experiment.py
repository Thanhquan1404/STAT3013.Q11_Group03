"""
Experiment with K-Folds validation for XGBoost, ROC curve, confusion matrix and export CSV.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from XGBoost_module import XGBoostClassifier
import os


class XGBoostExperiment:
    """
    XGBoost experiment với K-Folds Cross-Validation.
    """
    
    def __init__(self, eta=0.05, max_depth=6, subsample=0.8, 
                 colsample_bytree=0.8, num_boost_round=500, 
                 early_stopping_rounds=50, n_folds=5, verbose=True):
        """
        Initialize XGBoost experiment.
        """
        self.eta = eta
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.n_folds = n_folds
        self.verbose = verbose
        
        self.fold_results = []
        self.all_y_true = []
        self.all_y_pred = []
        self.all_y_proba = []
        self.classes_ = None
        self.feature_importance_list = []
        self.label_mapping = {}
        self.reverse_mapping = {}
    
    def run(self, X, y):
        """
        Run K-Folds experiment.
        """
        print("="*80)
        print(f"   XGBoost K-FOLDS EXPERIMENT | eta={self.eta}, depth={self.max_depth}, folds={self.n_folds}")
        print("="*80)
        
        self.classes_ = np.unique(y)
        
        self.label_mapping = {label: idx for idx, label in enumerate(self.classes_)}
        self.reverse_mapping = {idx: label for label, idx in self.label_mapping.items()}
        y_mapped = np.array([self.label_mapping[label] for label in y])
        
        print(f"\n📋 Label Mapping: {self.label_mapping}")
        print(f"   Original labels: {self.classes_}")
        print(f"   Mapped to: {list(self.reverse_mapping.keys())}")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_mapped), 1):
            print(f"\n{'='*80}")
            print(f"📊 Fold {fold}/{self.n_folds}")
            print(f"{'='*80}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_mapped[train_idx], y_mapped[val_idx]
            
            # Train model
            xgb_model = XGBoostClassifier(
                eta=self.eta,
                max_depth=self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                num_boost_round=self.num_boost_round,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=self.verbose
            )
            
            xgb_model.fit(X_train, y_train, X_val, y_val)
            
            # Predictions on validation set
            y_pred = xgb_model.predict(X_val)
            y_proba = xgb_model.predict_proba(X_val)
            
            # Calculate metrics (using mapped labels)
            metrics = self._calculate_metrics(y_val, y_pred, y_proba)
            metrics['fold'] = fold
            metrics['best_iteration'] = xgb_model.model.best_iteration if self.early_stopping_rounds else self.num_boost_round
            self.fold_results.append(metrics)
            
            # Store feature importance
            if xgb_model.model is not None:
                importance = xgb_model.model.get_score(importance_type='weight')
                self.feature_importance_list.append(importance)
            
            # Store for overall metrics (convert back to original labels)
            y_val_original = np.array([self.reverse_mapping[label] for label in y_val])
            y_pred_original = np.array([self.reverse_mapping[label] for label in y_pred])
            
            self.all_y_true.extend(y_val_original)
            self.all_y_pred.extend(y_pred_original)
            self.all_y_proba.extend(y_proba)
            
            # Print fold results
            print(f"\n{'─'*80}")
            print(f"📈 Fold {fold} Results:")
            print(f"{'─'*80}")
            print(f"   Accuracy:  {metrics['accuracy']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall:    {metrics['recall']:.4f}")
            print(f"   F1-Score:  {metrics['f1_score']:.4f}")
            if not np.isnan(metrics['roc_auc']):
                print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"   Best Iteration: {metrics['best_iteration']}")
        
        self._print_summary()
        
        return self
    
    def _calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate all metrics for one fold."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # ROC-AUC
        try:
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_proba, 
                    multi_class='ovr', 
                    average='weighted'
                )
        except:
            metrics['roc_auc'] = np.nan
        
        return metrics
    
    def _print_summary(self):
        """Print overall summary."""
        df = pd.DataFrame(self.fold_results)
        
        print("\n" + "="*80)
        print("                         SUMMARY RESULTS")
        print("="*80)
        
        print(f"\n{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-"*80)
        
        for col in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                min_val = df[col].min()
                max_val = df[col].max()
                
                print(f"{col.capitalize():<15} "
                      f"{mean_val:<12.4f} "
                      f"{std_val:<12.4f} "
                      f"{min_val:<12.4f} "
                      f"{max_val:<12.4f}")
        
        # Best iteration summary
        if 'best_iteration' in df.columns:
            print(f"{'Best Iteration':<15} "
                  f"{df['best_iteration'].mean():<12.1f} "
                  f"{df['best_iteration'].std():<12.1f} "
                  f"{df['best_iteration'].min():<12.0f} "
                  f"{df['best_iteration'].max():<12.0f}")
        
        print("="*80)
    
    def save_metrics_csv(self, filename='xgboost_metrics.csv'):
        """Save all metrics to CSV."""
        df = pd.DataFrame(self.fold_results)
        
        # Add summary row (mean)
        summary = {
            'fold': 'MEAN',
            'accuracy': df['accuracy'].mean(),
            'precision': df['precision'].mean(),
            'recall': df['recall'].mean(),
            'f1_score': df['f1_score'].mean(),
            'roc_auc': df['roc_auc'].mean(),
            'best_iteration': df['best_iteration'].mean()
        }
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
        
        # Add std row
        summary_std = {
            'fold': 'STD',
            'accuracy': df['accuracy'][:-1].std(),
            'precision': df['precision'][:-1].std(),
            'recall': df['recall'][:-1].std(),
            'f1_score': df['f1_score'][:-1].std(),
            'roc_auc': df['roc_auc'][:-1].std(),
            'best_iteration': df['best_iteration'][:-1].std()
        }
        df = pd.concat([df, pd.DataFrame([summary_std])], ignore_index=True)
        
        df.to_csv(filename, index=False)
        print(f"\n✔️ Metrics saved to: {filename}")
    
    def plot_confusion_matrix(self, filename='xgboost_confusion_matrix.png'):
        """Plot confusion matrix."""
        cm = confusion_matrix(self.all_y_true, self.all_y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=self.classes_, yticklabels=self.classes_)
        plt.title(f'Confusion Matrix - XGBoost (eta={self.eta}, depth={self.max_depth})', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✔️ Confusion matrix saved to: {filename}")
    
    def plot_roc_curve(self, filename='xgboost_roc_curve.png'):
        """Plot ROC curve."""
        y_true = np.array(self.all_y_true)
        y_proba = np.array(self.all_y_proba)
        
        plt.figure(figsize=(10, 8))
        
        if len(self.classes_) == 2:
            y_true_mapped = np.array([self.label_mapping[label] for label in y_true])
            pos_label = 1  
            
            fpr, tpr, _ = roc_curve(y_true_mapped, y_proba[:, 1], pos_label=pos_label)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})', color='green')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            
        else:
            # Multi-class classification
            y_true_mapped = np.array([self.label_mapping[label] for label in y_true])
            y_true_bin = label_binarize(y_true_mapped, classes=list(range(len(self.classes_))))
            colors = plt.cm.Set2(np.linspace(0, 1, len(self.classes_)))
            
            for i, (cls, color) in enumerate(zip(self.classes_, colors)):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, linewidth=2, color=color,
                        label=f'Class {cls} (AUC = {roc_auc:.4f})')
            
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - XGBoost (eta={self.eta}, depth={self.max_depth})', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✔️ ROC curve saved to: {filename}")
    
    def plot_feature_importance(self, feature_names=None, top_n=20, filename='xgboost_feature_importance.png'):
        """
        Plot average feature importance across all folds.
        """
        if not self.feature_importance_list:
            print("⚠️ No feature importance data available")
            return
        
        # Aggregate importance across folds
        all_features = set()
        for imp_dict in self.feature_importance_list:
            all_features.update(imp_dict.keys())
        
        # Calculate mean importance
        importance_mean = {}
        for feat in all_features:
            scores = [imp_dict.get(feat, 0) for imp_dict in self.feature_importance_list]
            importance_mean[feat] = np.mean(scores)
        
        # Sort and get top N
        sorted_importance = sorted(importance_mean.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_importance[:top_n]
        
        # Prepare data for plotting
        features = [f[0] for f in top_features]
        scores = [f[1] for f in top_features]
        
        # Map feature indices to names if provided
        if feature_names is not None:
            features = [feature_names[int(f[1:])] if f.startswith('f') else f for f in features]
        
        # Plot
        plt.figure(figsize=(10, max(6, len(features) * 0.3)))
        colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(features)))
        plt.barh(range(len(features)), scores, color=colors)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Average Importance Score', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance - XGBoost', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✔️ Feature importance plot saved to: {filename}")
