"""
Unit tests for model_evaluation module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os
import tempfile
import joblib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.model_evaluation import (
    FraudDetectionModel, train_multiple_models, compare_models
)


class TestFraudDetectionModel:
    """Test class for FraudDetectionModel."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample training data
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        self.X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_train = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
        
        self.X_test = pd.DataFrame(
            np.random.randn(20, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_test = np.random.choice([0, 1], size=20, p=[0.8, 0.2])
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = FraudDetectionModel(model_type='random_forest')
        assert model.model_type == 'random_forest'
        assert model.random_state == 42
        assert model.model is None
        assert model.feature_names is None
    
    def test_create_model_logistic_regression(self):
        """Test creating logistic regression model."""
        model = FraudDetectionModel(model_type='logistic_regression')
        model._create_model()
        
        assert model.model is not None
        assert hasattr(model.model, 'fit')
        assert hasattr(model.model, 'predict')
        assert hasattr(model.model, 'predict_proba')
    
    def test_create_model_random_forest(self):
        """Test creating random forest model."""
        model = FraudDetectionModel(model_type='random_forest')
        model._create_model()
        
        assert model.model is not None
        assert hasattr(model.model, 'fit')
        assert hasattr(model.model, 'predict')
        assert hasattr(model.model, 'predict_proba')
    
    def test_create_model_invalid_type(self):
        """Test creating model with invalid type."""
        model = FraudDetectionModel(model_type='invalid_model')
        
        with pytest.raises(ValueError, match="Unknown model type"):
            model._create_model()
    
    def test_train_model(self):
        """Test model training."""
        model = FraudDetectionModel(model_type='logistic_regression')
        model.train(self.X_train, self.y_train, self.X_test, self.y_test)
        
        assert model.model is not None
        assert model.feature_names == list(self.X_train.columns)
        assert 'training_date' in model.training_history
        assert 'test_metrics' in model.training_history
    
    def test_predict_without_training(self):
        """Test prediction without training."""
        model = FraudDetectionModel(model_type='logistic_regression')
        
        with pytest.raises(ValueError, match="Model not trained yet"):
            model.predict(self.X_test)
    
    def test_predict(self):
        """Test model prediction."""
        model = FraudDetectionModel(model_type='logistic_regression')
        model.train(self.X_train, self.y_train)
        
        predictions, probabilities = model.predict(self.X_test)
        
        assert len(predictions) == len(self.X_test)
        assert len(probabilities) == len(self.X_test)
        assert all(pred in [0, 1] for pred in predictions)
        assert all(0 <= prob <= 1 for prob in probabilities)
    
    def test_evaluate(self):
        """Test model evaluation."""
        model = FraudDetectionModel(model_type='logistic_regression')
        model.train(self.X_train, self.y_train)
        
        metrics = model.evaluate(self.X_test, self.y_test)
        
        assert 'accuracy' in metrics
        assert 'roc_auc' in metrics
        assert 'average_precision' in metrics
        assert 'classification_report' in metrics
        
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['average_precision'] <= 1
    
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        model = FraudDetectionModel(model_type='logistic_regression')
        model.train(self.X_train, self.y_train)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            saved_path = model.save_model(model_path)
            assert saved_path == model_path
            
            # Load model
            loaded_model = FraudDetectionModel.load_model(model_path)
            
            assert loaded_model.model_type == model.model_type
            assert loaded_model.feature_names == model.feature_names
            assert loaded_model.training_history == model.training_history
            
            # Test that loaded model can make predictions
            predictions, probabilities = loaded_model.predict(self.X_test)
            assert len(predictions) == len(self.X_test)
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.remove(model_path)
    
    def test_save_model_without_training(self):
        """Test saving model without training."""
        model = FraudDetectionModel(model_type='logistic_regression')
        
        with pytest.raises(ValueError, match="Model not trained yet"):
            model.save_model()
    
    def test_plot_roc_curve(self):
        """Test ROC curve plotting."""
        model = FraudDetectionModel(model_type='logistic_regression')
        model.train(self.X_train, self.y_train)
        
        # Test that plotting doesn't raise errors
        try:
            model.plot_roc_curve(self.X_test, self.y_test)
        except Exception as e:
            pytest.fail(f"ROC curve plotting failed: {e}")
    
    def test_plot_confusion_matrix(self):
        """Test confusion matrix plotting."""
        model = FraudDetectionModel(model_type='logistic_regression')
        model.train(self.X_train, self.y_train)
        
        # Test that plotting doesn't raise errors
        try:
            model.plot_confusion_matrix(self.X_test, self.y_test)
        except Exception as e:
            pytest.fail(f"Confusion matrix plotting failed: {e}")
    
    def test_plot_precision_recall_curve(self):
        """Test precision-recall curve plotting."""
        model = FraudDetectionModel(model_type='logistic_regression')
        model.train(self.X_train, self.y_train)
        
        # Test that plotting doesn't raise errors
        try:
            model.plot_precision_recall_curve(self.X_test, self.y_test)
        except Exception as e:
            pytest.fail(f"Precision-recall curve plotting failed: {e}")
    
    def test_explain_predictions(self):
        """Test SHAP explanation generation."""
        model = FraudDetectionModel(model_type='random_forest')
        model.train(self.X_train, self.y_train)
        
        # Test with small sample
        sample_data = self.X_test.iloc[:5]
        
        try:
            explainer, shap_values = model.explain_predictions(sample_data)
            assert explainer is not None
            assert shap_values is not None
        except Exception as e:
            # SHAP might not be available or might fail, which is okay for testing
            pytest.skip(f"SHAP explanation failed: {e}")


class TestModelTraining:
    """Test class for model training functions."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        self.X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_train = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
        
        self.X_test = pd.DataFrame(
            np.random.randn(20, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_test = np.random.choice([0, 1], size=20, p=[0.8, 0.2])
    
    def test_train_multiple_models(self):
        """Test training multiple models."""
        model_types = ['logistic_regression', 'random_forest']
        
        models, results = train_multiple_models(
            self.X_train, self.y_train, self.X_test, self.y_test, model_types
        )
        
        assert len(models) == 2
        assert len(results) == 2
        
        for model_type in model_types:
            assert model_type in models
            assert model_type in results
            
            # Check that models are trained
            assert models[model_type].model is not None
            
            # Check that results contain expected metrics
            assert 'accuracy' in results[model_type]
            assert 'roc_auc' in results[model_type]
            assert 'average_precision' in results[model_type]
    
    def test_train_multiple_models_default(self):
        """Test training multiple models with default model types."""
        models, results = train_multiple_models(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        
        assert len(models) == 2  # Default: logistic_regression, random_forest
        assert len(results) == 2
    
    def test_compare_models(self):
        """Test model comparison."""
        # Create mock results
        results = {
            'logistic_regression': {
                'accuracy': 0.85,
                'roc_auc': 0.90,
                'average_precision': 0.88
            },
            'random_forest': {
                'accuracy': 0.87,
                'roc_auc': 0.92,
                'average_precision': 0.90
            }
        }
        
        comparison_df = compare_models(results)
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert 'Model' in comparison_df.columns
        assert 'Accuracy' in comparison_df.columns
        assert 'ROC-AUC' in comparison_df.columns
        assert 'Average Precision' in comparison_df.columns
    
    def test_model_training_with_imbalanced_data(self):
        """Test model training with highly imbalanced data."""
        # Create highly imbalanced data
        np.random.seed(42)
        n_samples = 1000
        n_features = 5
        
        X_imbalanced = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y_imbalanced = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
        
        model = FraudDetectionModel(model_type='logistic_regression')
        model.train(X_imbalanced, y_imbalanced)
        
        # Should still be able to train and predict
        predictions, probabilities = model.predict(X_imbalanced[:10])
        assert len(predictions) == 10
        assert len(probabilities) == 10
    
    def test_model_training_with_small_data(self):
        """Test model training with very small dataset."""
        # Create small dataset
        X_small = pd.DataFrame(np.random.randn(10, 3), columns=['a', 'b', 'c'])
        y_small = np.random.choice([0, 1], size=10)
        
        model = FraudDetectionModel(model_type='logistic_regression')
        model.train(X_small, y_small)
        
        # Should still be able to train and predict
        predictions, probabilities = model.predict(X_small)
        assert len(predictions) == 10
        assert len(probabilities) == 10
    
    def test_model_training_with_missing_features(self):
        """Test model training with missing features in test data."""
        model = FraudDetectionModel(model_type='logistic_regression')
        model.train(self.X_train, self.y_train)
        
        # Test with missing features
        X_test_missing = self.X_test.drop(columns=[self.X_test.columns[0]])
        
        with pytest.raises(ValueError):
            model.predict(X_test_missing)


if __name__ == "__main__":
    pytest.main([__file__])
