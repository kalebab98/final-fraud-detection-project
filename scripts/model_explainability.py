# task3_model_explainability.ipynb

# 📦 Import libraries
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb
import joblib

# ⚙️ Load processed data
# Replace with your actual processed file path
fraud_data = pd.read_csv('data/Fraud_Data.csv')

# 🎯 Separate features and target
X = fraud_data.drop('class', axis=1)
y = fraud_data['class']

# 📊 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# List datetime columns that need conversion (customize if needed)
datetime_columns = ['signup_time', 'purchase_time']

# Convert only those
for col in datetime_columns:
    if col in X_train.columns:
        X_train[col] = pd.to_datetime(X_train[col], errors='coerce')
        X_test[col] = pd.to_datetime(X_test[col], errors='coerce')

        # Convert to numeric timestamp
        X_train[col] = X_train[col].astype('int64') // 10**9
        X_test[col] = X_test[col].astype('int64') // 10**9

# Now keep only numeric columns
X_train = X_train.select_dtypes(include=['number'])
X_test = X_test.select_dtypes(include=['number'])


# 🧪 Balance the data with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)




# 🔍 Scale the features
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

sample_size = 10000
if X_resampled.shape[0] > sample_size:
    indices = np.random.choice(len(X_resampled), sample_size, replace=False)
    X_resampled = X_resampled[indices]
    y_resampled = y_resampled.iloc[indices]

# 🧠 Train the final model (Random Forest)
#model = RandomForestClassifier(n_estimators=100, random_state=42)
#model.fit(X_resampled, y_resampled)
model = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0, use_label_encoder=False, eval_metric='logloss')
model.fit(X_resampled, y_resampled)
#model = HistGradientBoostingClassifier(random_state=42)
#model.fit(X_resampled, y_resampled)

# 💾 Save model if needed
# joblib.dump(model, 'models/final_random_forest.pkl')

# 🔎 SHAP Explainability
explainer = shap.Explainer(model, X_resampled)
shap_values = explainer(X_test_scaled[:100])  # Limit to first 100 for speed

# 📈 SHAP Summary Plot
shap.summary_plot(shap_values, X_test.iloc[:100])

# 📉 SHAP Bar Plot
shap.plots.bar(shap_values)

# 🔬 SHAP Force Plot for single prediction
sample_index = 0
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[sample_index].values, X_test.iloc[sample_index])


# 📌 SHAP Waterfall Plot
shap.plots.waterfall(shap_values[sample_index])
