"""
Model Comparison & Analysis Module
Subsystem 5: Model Training & Comparison

Comprehensive comparison between custom CNN and fine-tuned models.
Includes cross-validation, metrics analysis, and visual comparisons.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras


class ModelComparator:
    """Compare multiple trained models."""
    
    def __init__(self):
        """Initialize model comparator."""
        self.models = {}
        self.results = {}
        self.predictions = {}
        
    def add_model(self, model_path, model_name, model_type):
        """
        Add a model to comparison.
        
        Args:
            model_path: Path to saved model
            model_name: Name for identification
            model_type: 'custom_cnn', 'resnet50', or 'mobilenetv2'
        """
        model = keras.models.load_model(model_path)
        self.models[model_name] = {
            "model": model,
            "path": model_path,
            "type": model_type
        }
        print(f"✓ Loaded model: {model_name}")
    
    def evaluate_all_models(self, X_test, y_test, class_names=None):
        """
        Evaluate all models on test set.
        
        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            class_names: List of class names for reporting
        """
        results_df = pd.DataFrame()
        
        for model_name, model_info in self.models.items():
            print(f"\nEvaluating {model_name}...")
            model = model_info["model"]
            
            # Predictions
            y_pred_proba = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = np.argmax(y_test, axis=1)
            
            self.predictions[model_name] = {
                "y_pred": y_pred,
                "y_pred_proba": y_pred_proba,
                "y_true": y_true
            }
            
            # Metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            self.results[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "model_type": model_info["type"],
                "timestamp": str(datetime.now())
            }
            
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            
            results_df = pd.concat([results_df, pd.DataFrame({
                "Model": [model_name],
                "Accuracy": [accuracy],
                "Precision": [precision],
                "Recall": [recall],
                "F1-Score": [f1],
                "Type": [model_info["type"]]
            })], ignore_index=True)
        
        return results_df
    
    def plot_metrics_comparison(self, save_path=None):
        """Plot comparison of all metrics."""
        results_df = pd.DataFrame(self.results).T.reset_index()
        results_df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Type', 'Timestamp']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#1f77b4' if t == 'custom_cnn' else '#ff7f0e' if t == 'resnet50' else '#2ca02c' 
                 for t in results_df['Type']]
        
        for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
            ax.bar(results_df['Model'], results_df[metric], color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(results_df[metric]):
                ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison plot saved to {save_path}")
        
        plt.close()
        return results_df
    
    def plot_confusion_matrices(self, save_dir=None):
        """Plot confusion matrices for all models."""
        num_models = len(self.predictions)
        fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 5))
        
        if num_models == 1:
            axes = [axes]
        
        for ax, (model_name, preds) in zip(axes, self.predictions.items()):
            y_true = preds['y_true']
            y_pred = preds['y_pred']
            
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_title(f'{model_name}\nConfusion Matrix')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'confusion_matrices.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to {save_path}")
        
        plt.close()
    
    def plot_roc_curves(self, save_path=None):
        """Plot ROC curves for all models (binary or one-vs-rest)."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, preds in self.predictions.items():
            y_true = preds['y_true']
            y_pred_proba = preds['y_pred_proba']
            
            # For multiclass, compute average ROC
            n_classes = y_pred_proba.shape[1]
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            # Compute micro-average ROC
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Micro-average
            fpr_micro = np.interp(np.linspace(0, 1, 100), 
                                 np.concatenate([fpr[i] for i in range(n_classes)]),
                                 np.concatenate([tpr[i] for i in range(n_classes)]))
            tpr_micro = np.linspace(0, 1, 100)
            roc_auc_micro = np.mean(list(roc_auc.values()))
            
            ax.plot(fpr_micro, tpr_micro, linewidth=2, label=f'{model_name} (AUC = {roc_auc_micro:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        
        plt.close()
    
    def save_results_json(self, save_path):
        """Save detailed results to JSON."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert numpy types to native Python types
        results_serializable = {}
        for model_name, metrics in self.results.items():
            results_serializable[model_name] = {
                k: float(v) if isinstance(v, np.floating) else v 
                for k, v in metrics.items()
            }
        
        with open(save_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to {save_path}")
    
    def generate_comparison_report(self, save_path=None):
        """Generate comprehensive comparison report."""
        report = "=" * 80 + "\n"
        report += "MODEL COMPARISON REPORT\n"
        report += "=" * 80 + "\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Results table
        report += "PERFORMANCE METRICS:\n"
        report += "-" * 80 + "\n"
        results_df = pd.DataFrame(self.results).T.reset_index()
        report += results_df.to_string(index=False) + "\n\n"
        
        # Best models
        report += "BEST PERFORMERS:\n"
        report += "-" * 80 + "\n"
        results_df.set_index('index', inplace=True)
        # Normalize column names to match available columns
        available_metrics = [col for col in results_df.columns if col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']]
        for metric in available_metrics:
            try:
                best_model = results_df[metric].idxmax()
                best_value = results_df[metric].max()
                report += f"  Highest {metric}: {best_model} ({best_value:.4f})\n"
            except (KeyError, ValueError):
                report += f"  {metric}: N/A\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Comparison report saved to {save_path}")
        
        return report


def cross_validate_model(model_path, X, y, cv_folds=5):
    """
    Perform cross-validation on a model.
    
    Args:
        model_path: Path to saved model
        X: Feature data
        y: Target labels
        cv_folds: Number of cross-validation folds
    """
    model = keras.models.load_model(model_path)
    
    # Note: sklearn cross_val_score doesn't work directly with Keras models
    # This is a simplified implementation
    print(f"Cross-validating model with {cv_folds} folds...")
    
    kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    y_labels = np.argmax(y, axis=1)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y_labels)):
        print(f"  Fold {fold+1}/{cv_folds}...", end=' ')
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        # Clone model for this fold
        fold_model = keras.models.clone_model(model)
        fold_model.compile(model.optimizer, model.loss, model.metrics)
        fold_model.fit(X_fold_train, y_fold_train, epochs=5, batch_size=32, verbose=0)
        
        score = fold_model.evaluate(X_fold_val, y_fold_val, verbose=0)[1]
        cv_scores.append(score)
        print(f"Accuracy: {score:.4f}")
    
    print(f"\nCross-Validation Results:")
    print(f"  Mean Accuracy: {np.mean(cv_scores):.4f}")
    print(f"  Std Dev: {np.std(cv_scores):.4f}")
    
    return cv_scores


if __name__ == "__main__":
    print("=" * 80)
    print("MODEL COMPARISON MODULE - SUBSYSTEM 5")
    print("=" * 80)
