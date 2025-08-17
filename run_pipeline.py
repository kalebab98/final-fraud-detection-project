#!/usr/bin/env python3
"""
Fraud Detection Pipeline Runner

This script runs the complete fraud detection pipeline:
1. Data preprocessing
2. Model training
3. Model evaluation
4. Dashboard launch (optional)
"""

import os
import sys
import argparse
from datetime import datetime

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.data_cleaning import prepare_data_for_modeling
from scripts.model_evaluation import train_multiple_models, compare_models


def main():
    """Main pipeline execution function."""
    
    parser = argparse.ArgumentParser(description='Fraud Detection Pipeline')
    parser.add_argument('--data-dir', default='data', help='Directory containing data files')
    parser.add_argument('--models', nargs='+', 
                       default=['logistic_regression', 'random_forest'],
                       help='Model types to train')
    parser.add_argument('--dashboard', action='store_true', 
                       help='Launch dashboard after training')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training (use existing models)')
    
    args = parser.parse_args()
    
    print("🕵️ Fraud Detection Pipeline")
    print("=" * 50)
    
    # Check if data files exist
    required_files = [
        os.path.join(args.data_dir, 'Fraud_Data.csv'),
        os.path.join(args.data_dir, 'IpAddress_to_Country.csv'),
        os.path.join(args.data_dir, 'creditcard.csv')
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing required data files: {missing_files}")
        print("Please ensure all data files are present in the data/ directory.")
        return 1
    
    print("✅ Data files found")
    
    try:
        # Step 1: Data Preprocessing
        print("\n📊 Step 1: Data Preprocessing")
        print("-" * 30)
        
        start_time = datetime.now()
        X_train, y_train, X_test, y_test, feature_names = prepare_data_for_modeling(
            os.path.join(args.data_dir, 'Fraud_Data.csv'),
            os.path.join(args.data_dir, 'IpAddress_to_Country.csv'),
            os.path.join(args.data_dir, 'creditcard.csv')
        )
        
        preprocessing_time = datetime.now() - start_time
        print(f"✅ Data preprocessing completed in {preprocessing_time.total_seconds():.2f} seconds")
        print(f"   - Training samples: {len(X_train):,}")
        print(f"   - Test samples: {len(X_test):,}")
        print(f"   - Features: {len(feature_names)}")
        
        # Step 2: Model Training (if not skipped)
        if not args.skip_training:
            print("\n🤖 Step 2: Model Training")
            print("-" * 30)
            
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            start_time = datetime.now()
            models, results = train_multiple_models(
                X_train, y_train, X_test, y_test, args.models
            )
            
            training_time = datetime.now() - start_time
            print(f"✅ Model training completed in {training_time.total_seconds():.2f} seconds")
            
            # Step 3: Model Comparison
            print("\n📈 Step 3: Model Comparison")
            print("-" * 30)
            
            comparison_df = compare_models(results)
            
            # Save comparison results
            comparison_df.to_csv('model_comparison_results.csv', index=False)
            print("✅ Model comparison results saved to 'model_comparison_results.csv'")
            
        else:
            print("\n⏭️ Step 2: Model Training (Skipped)")
            print("-" * 30)
            print("Using existing models from models/ directory")
        
        # Step 4: Dashboard Launch (if requested)
        if args.dashboard:
            print("\n📊 Step 4: Launching Dashboard")
            print("-" * 30)
            print("🚀 Starting Streamlit dashboard...")
            print("   Dashboard will be available at: http://localhost:8501")
            print("   Press Ctrl+C to stop the dashboard")
            
            try:
                import subprocess
                subprocess.run(['streamlit', 'run', 'dashboard/app.py'])
            except KeyboardInterrupt:
                print("\n👋 Dashboard stopped by user")
            except FileNotFoundError:
                print("❌ Streamlit not found. Please install with: pip install streamlit")
                return 1
        
        print("\n🎉 Pipeline completed successfully!")
        print("=" * 50)
        
        # Summary
        print("\n📋 Summary:")
        print(f"   - Data preprocessing: {preprocessing_time.total_seconds():.2f}s")
        if not args.skip_training:
            print(f"   - Model training: {training_time.total_seconds():.2f}s")
            print(f"   - Models trained: {len(args.models)}")
            print(f"   - Best model: {comparison_df.loc[comparison_df['ROC-AUC'].idxmax(), 'Model']}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
