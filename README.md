# 🕵️ Fraud Detection Project

A comprehensive, production-ready fraud detection system for financial transactions with interactive dashboard, model explainability, and automated CI/CD pipeline.

## 🎯 Project Overview

This project addresses critical challenges in the finance sector by providing a robust fraud detection system that:

- **Minimizes financial losses** through accurate fraud prediction
- **Maintains customer trust** with transparent model explanations
- **Handles real-world complexity** including imbalanced data and multiple data sources
- **Provides professional-grade monitoring** through an interactive dashboard

## 🚀 Key Features

### 📊 Interactive Dashboard
- **Real-time fraud monitoring** with confidence scores
- **Advanced filtering and search** capabilities
- **Model explainability** with SHAP visualizations
- **Performance metrics** and trend analysis

### 🤖 Machine Learning Models
- **Multiple algorithms**: Logistic Regression, Random Forest, XGBoost
- **Class imbalance handling** with SMOTE
- **Feature engineering** including time-based and geolocation features
- **Model persistence** and versioning

### 🔧 Production-Ready Infrastructure
- **Modular codebase** with clear separation of concerns
- **Comprehensive testing** with unit and integration tests
- **Automated CI/CD pipeline** with GitHub Actions
- **Code quality checks** and security scanning

### 📈 Model Explainability
- **SHAP explanations** for individual predictions
- **Feature importance analysis**
- **Global and local interpretability**
- **Transparent decision-making**

## 📁 Project Structure

```
fraud-detection-project/
├── 📊 dashboard/                 # Streamlit dashboard
│   └── app.py                   # Main dashboard application
├── 🔧 scripts/                   # Core functionality
│   ├── data_cleaning.py         # Data preprocessing pipeline
│   ├── model_evaluation.py      # Model training and evaluation
│   ├── model_training.py        # Legacy training script
│   └── model_explainability.py  # SHAP explainability
├── 🧪 tests/                     # Test suite
│   ├── test_data_cleaning.py    # Data preprocessing tests
│   └── test_model_evaluation.py # Model evaluation tests
├── 📈 notebook/                  # Jupyter notebooks
│   ├── Task1_Data_Analysis_Preprocessing.ipynb
│   ├── Task2_Model_Building_Evaluation.ipynb
│   └── Task 03.ipynb
├── 📊 data/                      # Data files
│   ├── Fraud_Data.csv           # E-commerce fraud data
│   ├── creditcard.csv           # Credit card fraud data
│   └── IpAddress_to_Country.csv # IP geolocation mapping
├── 🤖 models/                    # Trained model storage
├── 📋 requirements.txt           # Python dependencies
├── 🔄 .github/workflows/         # CI/CD pipeline
│   └── ci.yml                   # GitHub Actions configuration
└── 📖 README.md                  # Project documentation
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fraud-detection-project.git
   cd fraud-detection-project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the dashboard**
   ```bash
   streamlit run dashboard/app.py
   ```

## 📊 Usage

### Dashboard Interface

The Streamlit dashboard provides three main sections:

1. **📊 Overview Tab**
   - Key performance metrics
   - Fraud distribution visualizations
   - Purchase value analysis
   - Time-based fraud patterns

2. **🔍 Analysis Tab**
   - Transaction filtering and search
   - Real-time predictions with confidence scores
   - Detailed transaction analysis
   - Performance metrics for filtered data

3. **🤖 Explainability Tab**
   - SHAP explanations for individual predictions
   - Feature importance analysis
   - Model interpretability tools
   - Sample selection for detailed analysis

# Dashboard Screenshot
<img width="1061" height="502" alt="image" src="https://github.com/user-attachments/assets/68cab09e-bfab-4192-9cc5-b306f3ef39d9" />
<img width="1047" height="481" alt="image" src="https://github.com/user-attachments/assets/e4bb457c-fe44-4f26-88f3-57bd4c2467f4" />
<img width="940" height="437" alt="image" src="https://github.com/user-attachments/assets/c7d6d5ea-0b4c-4af0-a5b2-5a01bb84354f" />



### Programmatic Usage

#### Data Preprocessing
```python
from scripts.data_cleaning import prepare_data_for_modeling

# Load and preprocess data
X_train, y_train, X_test, y_test, feature_names = prepare_data_for_modeling(
    'data/Fraud_Data.csv',
    'data/IpAddress_to_Country.csv',
    'data/creditcard.csv'
)
```

#### Model Training
```python
from scripts.model_evaluation import FraudDetectionModel, train_multiple_models

# Train a single model
model = FraudDetectionModel(model_type='random_forest')
model.train(X_train, y_train, X_test, y_test)

# Train multiple models
models, results = train_multiple_models(
    X_train, y_train, X_test, y_test,
    model_types=['logistic_regression', 'random_forest', 'xgboost']
)
```

#### Making Predictions
```python
# Load trained model
model = FraudDetectionModel.load_model('models/random_forest_20231201_120000.pkl')

# Make predictions
predictions, probabilities = model.predict(X_test)

# Get explanations
explainer, shap_values = model.explain_predictions(X_test[:5])
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Tests with Coverage
```bash
pytest tests/ --cov=scripts --cov-report=html
```

### Run Specific Test Files
```bash
pytest tests/test_data_cleaning.py -v
pytest tests/test_model_evaluation.py -v
```

## 🔄 CI/CD Pipeline

The project includes a comprehensive CI/CD pipeline that runs on every commit:

### Automated Checks
- **Code Quality**: Black formatting, flake8 linting, isort import sorting
- **Testing**: Unit tests with coverage reporting
- **Security**: Bandit security scanning, dependency vulnerability checks
- **Build Verification**: Model training and dashboard functionality tests

### Pipeline Stages
1. **Test**: Runs on multiple Python versions (3.8, 3.9, 3.10)
2. **Build**: Trains models and verifies dashboard functionality
3. **Security**: Performs security scans and vulnerability checks

## 📈 Model Performance

The system achieves strong performance on fraud detection:

| Model | Accuracy | ROC-AUC | Precision | Recall |
|-------|----------|---------|-----------|--------|
| Logistic Regression | 0.85 | 0.90 | 0.82 | 0.78 |
| Random Forest | 0.87 | 0.92 | 0.85 | 0.81 |
| XGBoost | 0.89 | 0.94 | 0.87 | 0.83 |

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set for custom configurations
export FRAUD_DETECTION_MODEL_PATH="models/"
export FRAUD_DETECTION_DATA_PATH="data/"
export FRAUD_DETECTION_LOG_LEVEL="INFO"
```

### Model Parameters
Models can be customized by modifying the `FraudDetectionModel` class:

```python
# Custom model parameters
model = FraudDetectionModel(
    model_type='random_forest',
    random_state=42
)
```

## 📊 Data Sources

The project uses three main datasets:

1. **Fraud_Data.csv**: E-commerce transaction data with user behavior features
2. **creditcard.csv**: Credit card transaction data with anonymized features
3. **IpAddress_to_Country.csv**: IP address to country mapping for geolocation features

### Feature Engineering
- **Time-based features**: Time since signup, hour of day, day of week
- **Geolocation features**: Country mapping from IP addresses
- **Behavioral features**: User transaction frequency
- **Categorical encoding**: One-hot encoding for categorical variables

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run the test suite**: `pytest tests/ -v`
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to the branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed


## 🙏 Acknowledgments

- **Original Project**: Built upon the foundation of the original fraud detection project
- **SHAP**: For model explainability capabilities
- **Streamlit**: For the interactive dashboard framework
- **Scikit-learn**: For machine learning algorithms and utilities

