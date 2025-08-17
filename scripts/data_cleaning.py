"""
data_cleaning.py

Data preprocessing and cleaning functions for fraud detection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")


def load_data(fraud_data_path, ip_data_path, credit_data_path):
    """
    Load all datasets for fraud detection.
    
    Args:
        fraud_data_path (str): Path to fraud data CSV
        ip_data_path (str): Path to IP address to country mapping CSV
        credit_data_path (str): Path to credit card data CSV
    
    Returns:
        tuple: (fraud_data, ip_data, credit_data)
    """
    fraud_data = pd.read_csv(fraud_data_path)
    ip_data = pd.read_csv(ip_data_path)
    credit_data = pd.read_csv(credit_data_path)
    
    return fraud_data, ip_data, credit_data


def clean_fraud_data(fraud_data):
    """
    Clean the fraud dataset by removing duplicates and converting data types.
    
    Args:
        fraud_data (pd.DataFrame): Raw fraud data
    
    Returns:
        pd.DataFrame: Cleaned fraud data
    """
    # Remove duplicates
    fraud_data = fraud_data.drop_duplicates()
    
    # Convert time columns to datetime
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
    
    # Handle missing values
    fraud_data = fraud_data[fraud_data['ip_address'].notnull()]
    
    return fraud_data


def add_ip_features(fraud_data, ip_data):
    """
    Add country information based on IP addresses.
    
    Args:
        fraud_data (pd.DataFrame): Fraud data with IP addresses
        ip_data (pd.DataFrame): IP to country mapping data
    
    Returns:
        pd.DataFrame: Fraud data with country information
    """
    # Convert ip_address to integer (handle IP addresses properly)
    def ip_to_int(ip_str):
        try:
            # For IP addresses like '192.168.1.1', convert to integer
            if isinstance(ip_str, str) and '.' in ip_str:
                parts = ip_str.split('.')
                return int(parts[0]) * 256**3 + int(parts[1]) * 256**2 + int(parts[2]) * 256 + int(parts[3])
            else:
                return int(float(ip_str))
        except:
            return 0
    
    fraud_data['ip_int'] = fraud_data['ip_address'].apply(ip_to_int)
    
    # Prepare ip_data for merging
    ip_data['lower'] = ip_data['lower_bound_ip_address'].astype(int)
    ip_data['upper'] = ip_data['upper_bound_ip_address'].astype(int)
    
    # Map each ip_int to a country
    def map_ip_to_country(ip):
        match = ip_data[(ip_data['lower'] <= ip) & (ip_data['upper'] >= ip)]
        return match['country'].values[0] if not match.empty else 'Unknown'
    
    fraud_data['country'] = fraud_data['ip_int'].apply(map_ip_to_country)
    
    return fraud_data


def feature_engineering(fraud_data):
    """
    Create engineered features for fraud detection.
    
    Args:
        fraud_data (pd.DataFrame): Fraud data with basic features
    
    Returns:
        pd.DataFrame: Fraud data with engineered features
    """
    # Time-based features
    fraud_data['time_since_signup'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds()
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
    
    # Transaction frequency feature
    tx_freq = fraud_data.groupby('user_id')['purchase_time'].count().rename('user_transaction_count')
    fraud_data = fraud_data.merge(tx_freq, on='user_id')
    
    return fraud_data


def encode_and_scale(fraud_data):
    """
    Encode categorical variables and scale numerical features.
    
    Args:
        fraud_data (pd.DataFrame): Fraud data with features
    
    Returns:
        pd.DataFrame: Fraud data with encoded and scaled features
    """
    # One-hot encoding for categorical variables
    categorical_cols = ['source', 'browser', 'sex', 'country']
    cols_to_encode = [col for col in categorical_cols if col in fraud_data.columns]
    
    if cols_to_encode:
        fraud_data = pd.get_dummies(fraud_data, columns=cols_to_encode, drop_first=True)
    
    # Scale numerical features
    scaler = StandardScaler()
    fraud_data['purchase_value'] = scaler.fit_transform(fraud_data[['purchase_value']])
    fraud_data['time_since_signup'] = scaler.fit_transform(fraud_data[['time_since_signup']])
    
    return fraud_data


def split_and_balance(fraud_data):
    """
    Split data into train/test sets and balance using SMOTE.
    
    Args:
        fraud_data (pd.DataFrame): Processed fraud data
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test) - balanced training data and test data
    """
    # Prepare features and target
    X = fraud_data.drop(columns=['class', 'signup_time', 'purchase_time', 'ip_address', 'device_id', 'user_id', 'ip_int'])
    y = fraud_data['class']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Apply SMOTE to balance training data
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    return X_train_bal, y_train_bal, X_test, y_test


def prepare_data_for_modeling(fraud_data_path, ip_data_path, credit_data_path):
    """
    Complete data preprocessing pipeline.
    
    Args:
        fraud_data_path (str): Path to fraud data CSV
        ip_data_path (str): Path to IP address to country mapping CSV
        credit_data_path (str): Path to credit card data CSV
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test, feature_names)
    """
    # Load data
    fraud_data, ip_data, credit_data = load_data(fraud_data_path, ip_data_path, credit_data_path)
    
    # Clean and preprocess
    fraud_data = clean_fraud_data(fraud_data)
    fraud_data = add_ip_features(fraud_data, ip_data)
    fraud_data = feature_engineering(fraud_data)
    fraud_data = encode_and_scale(fraud_data)
    
    # Split and balance
    X_train, y_train, X_test, y_test = split_and_balance(fraud_data)
    
    # Get feature names
    feature_names = X_train.columns.tolist()
    
    return X_train, y_train, X_test, y_test, feature_names
