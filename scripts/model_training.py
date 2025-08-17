"""
model_training.py

Simple model training and evaluation for fraud detection.
"""

from data_cleaning import load_data, clean_fraud_data, add_ip_features, feature_engineering, encode_and_scale, split_and_balance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


def plot_roc_curve(y_test, y_proba, label):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()


def plot_confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def main():
    # Load and preprocess data
    fraud_data, ip_data, credit_data = load_data(
        'data/Fraud_Data.csv',
        'data/IpAddress_to_Country.csv',
        'data/creditcard.csv'
    )
    fraud_data = clean_fraud_data(fraud_data)
    fraud_data = add_ip_features(fraud_data, ip_data)
    fraud_data = feature_engineering(fraud_data)
    fraud_data = encode_and_scale(fraud_data)
    X_train, y_train, X_test, y_test = split_and_balance(fraud_data)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]
    print("Logistic Regression Results:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
    print("Classification Report:\n", classification_report(y_test, y_pred_lr))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_proba_lr))
    plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression Confusion Matrix")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    print("\nRandom Forest Results:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
    print("Classification Report:\n", classification_report(y_test, y_pred_rf))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_proba_rf))
    plot_confusion_matrix(y_test, y_pred_rf, "Random Forest Confusion Matrix")

    # ROC Curve for both models
    plt.figure(figsize=(8, 6))
    plot_roc_curve(y_test, y_proba_lr, 'Logistic Regression')
    plot_roc_curve(y_test, y_proba_rf, 'Random Forest')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

# Note: If you get import errors, try running with:
# python -m scripts.model_training 