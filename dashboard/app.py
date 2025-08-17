"""
Fraud Detection Dashboard

An interactive Streamlit dashboard for fraud monitoring and model explainability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_cleaning import prepare_data_for_modeling
from scripts.model_evaluation import FraudDetectionModel

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🕵️",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load and cache the fraud detection data."""
    try:
        X_train, y_train, X_test, y_test, feature_names = prepare_data_for_modeling(
            'data/Fraud_Data.csv',
            'data/IpAddress_to_Country.csv',
            'data/creditcard.csv'
        )
        test_df = X_test.copy()
        test_df['actual_fraud'] = y_test
        return X_train, y_train, X_test, y_test, feature_names, test_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None, None

@st.cache_resource
def load_models():
    """Load trained models."""
    models = {}
    models_dir = "models"
    
    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            if filename.endswith('.pkl'):
                try:
                    model_path = os.path.join(models_dir, filename)
                    model = FraudDetectionModel.load_model(model_path)
                    models[model.model_type] = model
                except Exception as e:
                    st.warning(f"Could not load model {filename}: {str(e)}")
    return models

def main():
    """Main dashboard function."""
    
    st.title("🕵️ Fraud Detection Dashboard")
    
    # Load data and models
    with st.spinner("Loading data and models..."):
        X_train, y_train, X_test, y_test, feature_names, test_df = load_data()
        models = load_models()
    
    if test_df is None:
        st.error("Failed to load data. Please check your data files.")
        return
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # Model selection
    if models:
        selected_model_name = st.sidebar.selectbox(
            "Select Model",
            list(models.keys()),
            index=0
        )
        selected_model = models[selected_model_name]
    else:
        st.sidebar.warning("No trained models found.")
        selected_model = None
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Analysis", "🤖 Explainability"])
    
    with tab1:
        show_overview_tab(test_df, selected_model)
    
    with tab2:
        show_analysis_tab(test_df, selected_model)
    
    with tab3:
        show_explainability_tab(test_df, selected_model)

def show_overview_tab(test_df, selected_model):
    """Show overview tab."""
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{len(test_df):,}")
    
    with col2:
        fraud_count = test_df['actual_fraud'].sum()
        fraud_rate = (fraud_count / len(test_df)) * 100
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%", f"{fraud_count:,} transactions")
    
    with col3:
        if selected_model:
            predictions, probabilities = selected_model.predict(test_df.drop('actual_fraud', axis=1))
            predicted_fraud = predictions.sum()
            predicted_fraud_rate = (predicted_fraud / len(test_df)) * 100
            st.metric("Predicted Fraud Rate", f"{predicted_fraud_rate:.2f}%")
        else:
            st.metric("Predicted Fraud Rate", "N/A")
    
    with col4:
        if selected_model:
            accuracy = (predictions == test_df['actual_fraud']).mean() * 100
            st.metric("Model Accuracy", f"{accuracy:.2f}%")
        else:
            st.metric("Model Accuracy", "N/A")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fraud_counts = test_df['actual_fraud'].value_counts()
        fig = px.pie(
            values=fraud_counts.values,
            names=['Legitimate', 'Fraudulent'],
            title="Actual Fraud Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            test_df,
            x='purchase_value',
            color='actual_fraud',
            title="Purchase Value by Fraud Status"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_analysis_tab(test_df, selected_model):
    """Show analysis tab."""
    st.header("🔍 Transaction Analysis")
    
    # Add predictions if model is available
    if selected_model:
        predictions, probabilities = selected_model.predict(test_df.drop('actual_fraud', axis=1))
        analysis_df = test_df.copy()
        analysis_df['predicted_fraud'] = predictions
        analysis_df['fraud_probability'] = probabilities
    else:
        analysis_df = test_df.copy()
        analysis_df['predicted_fraud'] = np.nan
        analysis_df['fraud_probability'] = np.nan
    
    # Filters
    st.subheader("Filters")
    col1, col2 = st.columns(2)
    
    with col1:
        min_value = float(analysis_df['purchase_value'].min())
        max_value = float(analysis_df['purchase_value'].max())
        value_range = st.slider(
            "Purchase Value Range",
            min_value=min_value,
            max_value=max_value,
            value=(min_value, max_value)
        )
    
    with col2:
        fraud_filter = st.selectbox(
            "Fraud Status Filter",
            ["All", "Legitimate Only", "Fraudulent Only", "Predicted Fraud"]
        )
    
    # Apply filters
    filtered_df = analysis_df[
        (analysis_df['purchase_value'] >= value_range[0]) &
        (analysis_df['purchase_value'] <= value_range[1])
    ]
    
    if fraud_filter == "Legitimate Only":
        filtered_df = filtered_df[filtered_df['actual_fraud'] == 0]
    elif fraud_filter == "Fraudulent Only":
        filtered_df = filtered_df[filtered_df['actual_fraud'] == 1]
    elif fraud_filter == "Predicted Fraud" and selected_model:
        filtered_df = filtered_df[filtered_df['predicted_fraud'] == 1]
    
    # Display results
    st.subheader(f"Filtered Results ({len(filtered_df):,} transactions)")
    
    if len(filtered_df) > 0:
        # Show key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            actual_fraud_rate = (filtered_df['actual_fraud'].sum() / len(filtered_df)) * 100
            st.metric("Actual Fraud Rate", f"{actual_fraud_rate:.2f}%")
        
        with col2:
            if selected_model:
                predicted_fraud_rate = (filtered_df['predicted_fraud'].sum() / len(filtered_df)) * 100
                st.metric("Predicted Fraud Rate", f"{predicted_fraud_rate:.2f}%")
            else:
                st.metric("Predicted Fraud Rate", "N/A")
        
        with col3:
            avg_value = filtered_df['purchase_value'].mean()
            st.metric("Average Purchase Value", f"${avg_value:.2f}")
        
        # Display table
        display_cols = ['purchase_value', 'hour_of_day', 'day_of_week', 'user_transaction_count', 'age']
        if selected_model:
            display_cols.extend(['predicted_fraud', 'fraud_probability'])
        display_cols.append('actual_fraud')
        
        display_df = filtered_df[display_cols].copy()
        display_df['purchase_value'] = display_df['purchase_value'].round(2)
        if selected_model:
            display_df['fraud_probability'] = display_df['fraud_probability'].round(3)
        
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No transactions match the selected filters.")

def show_explainability_tab(test_df, selected_model):
    """Show explainability tab."""
    st.header("🤖 Model Explainability")
    
    if not selected_model:
        st.warning("Please select a trained model to view explainability features.")
        return
    
    # Sample selection
    st.subheader("Select Sample for Explanation")
    
    sample_method = st.radio(
        "Sample Selection Method",
        ["Random Sample", "High Fraud Probability", "Low Fraud Probability"]
    )
    
    # Get sample data
    if sample_method == "Random Sample":
        sample_index = np.random.randint(0, len(test_df))
    elif sample_method in ["High Fraud Probability", "Low Fraud Probability"]:
        predictions, probabilities = selected_model.predict(test_df.drop('actual_fraud', axis=1))
        if sample_method == "High Fraud Probability":
            sample_index = np.argmax(probabilities)
        else:
            sample_index = np.argmin(probabilities)
    
    sample_data = test_df.drop('actual_fraud', axis=1).iloc[[sample_index]]
    actual_label = test_df.iloc[sample_index]['actual_fraud']
    
    # Make prediction
    prediction, probability = selected_model.predict(sample_data)
    
    # Display prediction results
    st.subheader("Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prediction_text = "🟢 Legitimate" if prediction[0] == 0 else "🔴 Fraudulent"
        st.metric("Prediction", prediction_text)
    
    with col2:
        confidence = abs(probability[0] - 0.5) * 2
        st.metric("Confidence", f"{confidence:.1%}")
    
    with col3:
        actual_text = "🟢 Legitimate" if actual_label == 0 else "🔴 Fraudulent"
        st.metric("Actual", actual_text)
    
    # Feature importance
    st.subheader("Feature Importance")
    
    if hasattr(selected_model.model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': selected_model.feature_names,
            'Importance': selected_model.model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df.tail(10),
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 10 Most Important Features"
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
