#!/usr/bin/env python3
"""
Fraud Detection Demo

This script demonstrates the fraud detection pipeline with a small sample of data.
"""

import os
import sys
import time
from datetime import datetime

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.data_cleaning import prepare_data_for_modeling
from scripts.model_evaluation import FraudDetectionModel, train_multiple_models, compare_models


def main():
    """Run the fraud detection demo."""
    
    print("🕵️ Fraud Detection Demo")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Data Preprocessing
        print("\n📊 Step 1: Data Preprocessing")
        print("-" * 30)
        
        start_time = time.time()
        X_train, y_train, X_test, y_test, feature_names = prepare_data_for_modeling(
            'data/Fraud_Data.csv',
            'data/IpAddress_to_Country.csv',
            'data/creditcard.csv'
        )
        
        preprocessing_time = time.time() - start_time
        print(f"✅ Data preprocessing completed in {preprocessing_time:.2f} seconds")
        print(f"   - Training samples: {len(X_train):,}")
        print(f"   - Test samples: {len(X_test):,}")
        print(f"   - Features: {len(feature_names)}")
        print(f"   - Fraud rate in training: {(y_train == 1).mean():.2%}")
        print(f"   - Fraud rate in test: {(y_test == 1).mean():.2%}")
        
        # Step 2: Model Training (with smaller sample for demo)
        print("\n🤖 Step 2: Model Training")
        print("-" * 30)
        
        # Use a smaller sample for faster demo
        sample_size = min(5000, len(X_train))
        X_train_sample = X_train.sample(n=sample_size, random_state=42)
        y_train_sample = y_train.loc[X_train_sample.index]
        
        print(f"Using {sample_size:,} samples for training (for faster demo)")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        start_time = time.time()
        models, results = train_multiple_models(
            X_train_sample, y_train_sample, X_test, y_test,
            model_types=['logistic_regression', 'random_forest']
        )
        
        training_time = time.time() - start_time
        print(f"✅ Model training completed in {training_time:.2f} seconds")
        
        # Step 3: Model Comparison
        print("\n📈 Step 3: Model Comparison")
        print("-" * 30)
        
        comparison_df = compare_models(results)
        
        # Save comparison results
        comparison_df.to_csv('demo_model_comparison_results.csv', index=False)
        print("✅ Model comparison results saved to 'demo_model_comparison_results.csv'")
        
        # Step 4: Demo Predictions
        print("\n🔮 Step 4: Demo Predictions")
        print("-" * 30)
        
        # Get the best model
        best_model_name = comparison_df.loc[comparison_df['ROC-AUC'].idxmax(), 'Model']
        best_model = models[best_model_name]
        
        print(f"Using best model: {best_model_name}")
        
        # Make predictions on a small sample
        sample_size = min(100, len(X_test))
        X_test_sample = X_test.sample(n=sample_size, random_state=42)
        y_test_sample = y_test.loc[X_test_sample.index]
        
        predictions, probabilities = best_model.predict(X_test_sample)
        
        # Calculate metrics
        accuracy = (predictions == y_test_sample).mean()
        fraud_predictions = predictions.sum()
        actual_fraud = y_test_sample.sum()
        
        print(f"Sample predictions ({sample_size} transactions):")
        print(f"   - Accuracy: {accuracy:.2%}")
        print(f"   - Predicted fraud: {fraud_predictions}")
        print(f"   - Actual fraud: {actual_fraud}")
        print(f"   - Average fraud probability: {probabilities.mean():.3f}")
        
        # Show some example predictions
        print("\nExample predictions:")
        for i in range(min(5, len(X_test_sample))):
            pred = "🟢 Legitimate" if predictions[i] == 0 else "🔴 Fraudulent"
            actual = "🟢 Legitimate" if y_test_sample.iloc[i] == 0 else "🔴 Fraudulent"
            prob = probabilities[i]
            print(f"   Transaction {i+1}: {pred} (prob: {prob:.3f}) | Actual: {actual}")
        
        print("\n🎉 Demo completed successfully!")
        print("=" * 50)
        
        # Summary
        print("\n📋 Summary:")
        print(f"   - Data preprocessing: {preprocessing_time:.2f}s")
        print(f"   - Model training: {training_time:.2f}s")
        print(f"   - Best model: {best_model_name}")
        print(f"   - Best ROC-AUC: {comparison_df['ROC-AUC'].max():.3f}")
        
        print("\n🚀 Next steps:")
        print("   - Run 'streamlit run dashboard/app.py' to launch the dashboard")
        print("   - Run 'python run_pipeline.py --dashboard' for full pipeline")
        print("   - Check 'demo_model_comparison_results.csv' for detailed results")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
