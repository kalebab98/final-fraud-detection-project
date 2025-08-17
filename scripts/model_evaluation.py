"""
model_evaluation.py

Model training, evaluation, and saving functionality for fraud detection.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")


class FraudDetectionModel:
    """A class to handle fraud detection model training and evaluation."""
    
    def __init__(self, model_type='random_forest', random_state=42):
        """
        Initialize the fraud detection model.
        
        Args:
            model_type (str): Type of model ('logistic_regression', 'random_forest', 'xgboost')
            random_state (int): Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.training_history = {}
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
    def _create_model(self):
        """Create the specified model."""
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000, 
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    eval_metric='logloss',
                    use_label_encoder=False
                )
            except ImportError:
                print("XGBoost not available, using Random Forest instead")
                self.model_type = 'random_forest'
                self._create_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train, X_test=None, y_test=None):
        """
        Train the model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            X_test (pd.DataFrame): Test features (optional)
            y_test (pd.Series): Test labels (optional)
        """
        # Create model if not exists
        if self.model is None:
            self._create_model()
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Train the model
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set if provided
        if X_test is not None and y_test is not None:
            self.evaluate(X_test, y_test)
        
        # Store training history
        self.training_history = {
            'model_type': self.model_type,
            'training_date': datetime.now().isoformat(),
            'n_features': len(self.feature_names),
            'n_training_samples': len(X_train),
            'feature_names': self.feature_names,
            'test_metrics': None  # Will be populated if test data is provided
        }
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            tuple: (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Make predictions
        y_pred, y_proba = self.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': (y_pred == y_test).mean(),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'average_precision': average_precision_score(y_test, y_proba),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Store metrics in training history
        self.training_history['test_metrics'] = metrics
        
        # Print results
        print(f"\n{self.model_type.upper()} Model Evaluation:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Average Precision: {metrics['average_precision']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return metrics
    
    def plot_roc_curve(self, X_test, y_test, save_path=None):
        """Plot ROC curve."""
        y_pred, y_proba = self.predict(X_test)
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{self.model_type} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, X_test, y_test, save_path=None):
        """Plot confusion matrix."""
        y_pred, _ = self.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, X_test, y_test, save_path=None):
        """Plot precision-recall curve."""
        y_pred, y_proba = self.predict(X_test)
        
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'AP = {avg_precision:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def explain_predictions(self, X_sample, sample_names=None, max_display=10):
        """
        Generate SHAP explanations for predictions.
        
        Args:
            X_sample (pd.DataFrame): Sample data to explain
            sample_names (list): Names for the samples
            max_display (int): Maximum number of features to display
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Create SHAP explainer
        explainer = shap.Explainer(self.model, X_sample)
        shap_values = explainer(X_sample)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
        plt.title(f'SHAP Summary Plot - {self.model_type}')
        plt.tight_layout()
        plt.show()
        
        # Bar plot
        plt.figure(figsize=(10, 8))
        shap.plots.bar(shap_values, max_display=max_display, show=False)
        plt.title(f'SHAP Feature Importance - {self.model_type}')
        plt.tight_layout()
        plt.show()
        
        return explainer, shap_values
    
    def save_model(self, filepath=None):
        """
        Save the trained model and metadata.
        
        Args:
            filepath (str): Path to save the model (optional)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"models/{self.model_type}_{timestamp}.pkl"
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")
        
        return filepath
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            FraudDetectionModel: Loaded model instance
        """
        model_data = joblib.load(filepath)
        
        # Create instance
        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.training_history = model_data['training_history']
        
        return instance


def train_multiple_models(X_train, y_train, X_test, y_test, model_types=None):
    """
    Train and evaluate multiple models.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        model_types (list): List of model types to train
        
    Returns:
        dict: Dictionary of trained models and their metrics
    """
    if model_types is None:
        model_types = ['logistic_regression', 'random_forest']
    
    models = {}
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Training {model_type}")
        print(f"{'='*50}")
        
        # Create and train model
        model = FraudDetectionModel(model_type=model_type)
        model.train(X_train, y_train, X_test, y_test)
        
        # Store model and results
        models[model_type] = model
        # Get metrics from the evaluate method result
        if hasattr(model, 'training_history') and 'test_metrics' in model.training_history:
            results[model_type] = model.training_history['test_metrics']
        else:
            # If no metrics in training history, evaluate now
            metrics = model.evaluate(X_test, y_test)
            results[model_type] = metrics
        
        # Save model
        model.save_model()
    
    return models, results


def compare_models(results):
    """
    Compare multiple models and create comparison plots.
    
    Args:
        results (dict): Dictionary of model results
    """
    # Create comparison DataFrame
    comparison_data = []
    for model_name, metrics in results.items():
        if metrics is not None and isinstance(metrics, dict):
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0.0),
                'ROC-AUC': metrics.get('roc_auc', 0.0),
                'Average Precision': metrics.get('average_precision', 0.0)
            })
        else:
            print(f"Warning: No metrics available for {model_name}")
            comparison_data.append({
                'Model': model_name,
                'Accuracy': 0.0,
                'ROC-AUC': 0.0,
                'Average Precision': 0.0
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['Accuracy', 'ROC-AUC', 'Average Precision']
    for i, metric in enumerate(metrics):
        axes[i].bar(comparison_df['Model'], comparison_df[metric])
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison table
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df
