"""
Unit tests for data_cleaning module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_cleaning import (
    load_data, clean_fraud_data, add_ip_features, 
    feature_engineering, encode_and_scale, split_and_balance,
    prepare_data_for_modeling
)


class TestDataCleaning:
    """Test class for data cleaning functions."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample fraud data
        self.sample_fraud_data = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'signup_time': ['2023-01-01 10:00:00', '2023-01-01 11:00:00', 
                           '2023-01-01 12:00:00', '2023-01-01 13:00:00', 
                           '2023-01-01 14:00:00'],
            'purchase_time': ['2023-01-01 10:30:00', '2023-01-01 11:30:00', 
                             '2023-01-01 12:30:00', '2023-01-01 13:30:00', 
                             '2023-01-01 14:30:00'],
            'purchase_value': [100, 200, 300, 400, 500],
            'device_id': ['dev1', 'dev2', 'dev3', 'dev4', 'dev5'],
            'source': ['Ads', 'Direct', 'SEO', 'Ads', 'Direct'],
            'browser': ['Chrome', 'Firefox', 'Safari', 'Chrome', 'Firefox'],
            'sex': ['M', 'F', 'M', 'F', 'M'],
            'age': [25, 30, 35, 40, 45],
            'ip_address': ['192.168.1.1', '192.168.1.2', '192.168.1.3', 
                          '192.168.1.4', '192.168.1.5'],
            'class': [0, 1, 0, 1, 0]
        })
        
        # Create sample IP data
        self.sample_ip_data = pd.DataFrame({
            'lower_bound_ip_address': [3232235777, 3232235778, 3232235779, 3232235780, 3232235781],
            'upper_bound_ip_address': [3232235777, 3232235778, 3232235779, 3232235780, 3232235781],
            'country': ['United States', 'Canada', 'United Kingdom', 'Germany', 'France']
        })
        
        # Create sample credit data
        self.sample_credit_data = pd.DataFrame({
            'Time': [1, 2, 3, 4, 5],
            'V1': [0.1, 0.2, 0.3, 0.4, 0.5],
            'V2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'Amount': [100, 200, 300, 400, 500],
            'Class': [0, 1, 0, 1, 0]
        })
    
    def test_load_data(self):
        """Test load_data function."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = [
                self.sample_fraud_data,
                self.sample_ip_data,
                self.sample_credit_data
            ]
            
            fraud_data, ip_data, credit_data = load_data(
                'fake_fraud_path.csv',
                'fake_ip_path.csv',
                'fake_credit_path.csv'
            )
            
            assert isinstance(fraud_data, pd.DataFrame)
            assert isinstance(ip_data, pd.DataFrame)
            assert isinstance(credit_data, pd.DataFrame)
            assert len(fraud_data) == 5
            assert len(ip_data) == 5
            assert len(credit_data) == 5
    
    def test_clean_fraud_data(self):
        """Test clean_fraud_data function."""
        # Add a duplicate row
        fraud_data_with_duplicate = self.sample_fraud_data.copy()
        fraud_data_with_duplicate = pd.concat([fraud_data_with_duplicate, 
                                             fraud_data_with_duplicate.iloc[0:1]])
        
        cleaned_data = clean_fraud_data(fraud_data_with_duplicate)
        
        # Check that duplicates are removed
        assert len(cleaned_data) == len(self.sample_fraud_data)
        
        # Check that datetime columns are converted
        assert pd.api.types.is_datetime64_any_dtype(cleaned_data['signup_time'])
        assert pd.api.types.is_datetime64_any_dtype(cleaned_data['purchase_time'])
    
    def test_add_ip_features(self):
        """Test add_ip_features function."""
        fraud_data = clean_fraud_data(self.sample_fraud_data)
        enhanced_data = add_ip_features(fraud_data, self.sample_ip_data)
        
        # Check that country column is added
        assert 'country' in enhanced_data.columns
        assert 'ip_int' in enhanced_data.columns
        
        # Check that IP addresses are converted to integers
        assert enhanced_data['ip_int'].dtype == 'int64'
    
    def test_feature_engineering(self):
        """Test feature_engineering function."""
        fraud_data = clean_fraud_data(self.sample_fraud_data)
        fraud_data = add_ip_features(fraud_data, self.sample_ip_data)
        engineered_data = feature_engineering(fraud_data)
        
        # Check that new features are created
        assert 'time_since_signup' in engineered_data.columns
        assert 'hour_of_day' in engineered_data.columns
        assert 'day_of_week' in engineered_data.columns
        assert 'user_transaction_count' in engineered_data.columns
        
        # Check that time_since_signup is positive
        assert (engineered_data['time_since_signup'] >= 0).all()
        
        # Check that hour_of_day is between 0 and 23
        assert (engineered_data['hour_of_day'] >= 0).all()
        assert (engineered_data['hour_of_day'] <= 23).all()
        
        # Check that day_of_week is between 0 and 6
        assert (engineered_data['day_of_week'] >= 0).all()
        assert (engineered_data['day_of_week'] <= 6).all()
    
    def test_encode_and_scale(self):
        """Test encode_and_scale function."""
        fraud_data = clean_fraud_data(self.sample_fraud_data)
        fraud_data = add_ip_features(fraud_data, self.sample_ip_data)
        fraud_data = feature_engineering(fraud_data)
        encoded_data = encode_and_scale(fraud_data)
        
        # Check that categorical columns are encoded
        assert 'source_Ads' in encoded_data.columns
        assert 'source_Direct' in encoded_data.columns
        assert 'source_SEO' in encoded_data.columns
        assert 'browser_Chrome' in encoded_data.columns
        assert 'browser_Firefox' in encoded_data.columns
        assert 'browser_Safari' in encoded_data.columns
        assert 'sex_M' in encoded_data.columns
        assert 'country_United States' in encoded_data.columns
        
        # Check that numerical columns are scaled
        assert 'purchase_value' in encoded_data.columns
        assert 'time_since_signup' in encoded_data.columns
    
    def test_split_and_balance(self):
        """Test split_and_balance function."""
        fraud_data = clean_fraud_data(self.sample_fraud_data)
        fraud_data = add_ip_features(fraud_data, self.sample_ip_data)
        fraud_data = feature_engineering(fraud_data)
        fraud_data = encode_and_scale(fraud_data)
        
        X_train, y_train, X_test, y_test = split_and_balance(fraud_data)
        
        # Check that data is split correctly
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)
        
        # Check that target column is not in features
        assert 'class' not in X_train.columns
        assert 'class' not in X_test.columns
        
        # Check that SMOTE is applied (balanced classes in training)
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        assert len(unique_train) == 2
        assert abs(counts_train[0] - counts_train[1]) <= 1  # Should be balanced
    
    def test_prepare_data_for_modeling(self):
        """Test the complete data preprocessing pipeline."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = [
                self.sample_fraud_data,
                self.sample_ip_data,
                self.sample_credit_data
            ]
            
            X_train, y_train, X_test, y_test, feature_names = prepare_data_for_modeling(
                'fake_fraud_path.csv',
                'fake_ip_path.csv',
                'fake_credit_path.csv'
            )
            
            # Check that all expected outputs are returned
            assert isinstance(X_train, pd.DataFrame)
            assert isinstance(y_train, pd.Series)
            assert isinstance(X_test, pd.DataFrame)
            assert isinstance(y_test, pd.Series)
            assert isinstance(feature_names, list)
            
            # Check that feature names match training data
            assert list(X_train.columns) == feature_names
            
            # Check that data is properly shaped
            assert len(X_train) > 0
            assert len(X_test) > 0
            assert len(y_train) == len(X_train)
            assert len(y_test) == len(X_test)
    
    def test_error_handling(self):
        """Test error handling in data loading."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = FileNotFoundError("File not found")
            
            with pytest.raises(FileNotFoundError):
                load_data('nonexistent_file.csv', 'nonexistent_file.csv', 'nonexistent_file.csv')
    
    def test_missing_values_handling(self):
        """Test handling of missing values."""
        # Create data with missing values
        fraud_data_with_nulls = self.sample_fraud_data.copy()
        fraud_data_with_nulls.loc[0, 'ip_address'] = None
        
        cleaned_data = clean_fraud_data(fraud_data_with_nulls)
        
        # Check that rows with missing IP addresses are removed
        assert len(cleaned_data) == len(self.sample_fraud_data) - 1
        assert cleaned_data['ip_address'].notna().all()
    
    def test_feature_engineering_edge_cases(self):
        """Test feature engineering with edge cases."""
        # Test with single user
        single_user_data = self.sample_fraud_data[self.sample_fraud_data['user_id'] == 1].copy()
        single_user_data = clean_fraud_data(single_user_data)
        single_user_data = add_ip_features(single_user_data, self.sample_ip_data)
        
        engineered_data = feature_engineering(single_user_data)
        
        # Should still work with single user
        assert 'user_transaction_count' in engineered_data.columns
        assert engineered_data['user_transaction_count'].iloc[0] == 1


if __name__ == "__main__":
    pytest.main([__file__])
