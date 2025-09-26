# ==========================================
# Fraud Detection ML Pipeline
# ==========================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, confusion_matrix
)

from scipy.stats import uniform, randint
import joblib


# ==========================================
# 2. Load & Explore Data
# ==========================================
df = pd.read_csv("C:/Users/Winni/.kaggle/Fraud.csv")

print("Dataset Shape:", df.shape)
print(df['isFraud'].value_counts(normalize=True) * 100)
print(df['isFlaggedFraud'].value_counts(normalize=True) * 100)

# Keep original system predictions aside
fraud_flag = df['isFlaggedFraud']


# ==========================================
# 3. Preprocessing & Feature Engineering
# ==========================================
df['transaction_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['transfer_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
df['high_risk_type'] = df['type'].apply(lambda x: 1 if x in ['TRANSFER','CASH_OUT'] else 0)

# Drop unnecessary columns
df = df.drop(columns=['nameOrig','nameDest'])

# Encode transaction type
type_map = {'CASH_IN':0, 'CASH_OUT':1, 'DEBIT':2,'PAYMENT':3,'TRANSFER':4}
df['type'] = df['type'].map(type_map)

# Define features/target
X = df.drop(columns=['isFraud','isFlaggedFraud'])
y = df['isFraud']


# ==========================================
# 4. Train/Test Split & Handle Imbalance
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

print("Original train shape:", y_train.value_counts())
print("Resampled train shape:", y_res.value_counts())


# ==========================================
# 5. Hyperparameter Tuning (Subset)
# ==========================================
X_train_sample, _, y_train_sample, _ = train_test_split(
    X_res, y_res, train_size=0.2, stratify=y_res, random_state=42
)

param_grid_logreg = {
    'C': uniform(0.01, 10),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

param_grid_rf = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20)
}

param_grid_xgb = {
    'n_estimators': randint(50, 500),
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(3, 20),
    'subsample': uniform(0.5, 0.9),
    'colsample_bytree': uniform(0.5, 0.9)
}

# Initialize models
rf = RandomForestClassifier(random_state=42)
logreg = LogisticRegression(random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Randomized Search
random_searches = {
    "Random Forest": RandomizedSearchCV(rf, param_grid_rf, n_iter=30, cv=5, scoring='f1', random_state=42),
    "Logistic Regression": RandomizedSearchCV(logreg, param_grid_logreg, n_iter=30, cv=5, scoring='f1', random_state=42),
    "XGBoost": RandomizedSearchCV(xgb, param_grid_xgb, n_iter=30, cv=5, scoring='f1', random_state=42)
}

best_params = {}
for name, search in random_searches.items():
    search.fit(X_train_sample, y_train_sample)
    best_params[name] = search.best_params_
    print(f"{name} Best Params:", search.best_params_)


# ==========================================
# 6. Train Final Models on Full Resampled Data
# ==========================================
final_models = {
    "Random Forest": RandomForestClassifier(**best_params["Random Forest"], random_state=42),
    "Logistic Regression": LogisticRegression(**best_params["Logistic Regression"], random_state=42),
    "XGBoost": XGBClassifier(**best_params["XGBoost"], use_label_encoder=False, eval_metric='logloss', random_state=42)
}

for name, model in final_models.items():
    model.fit(X_res, y_res)
    final_models[name] = model


# ==========================================
# 7. Evaluation Helpers
# ==========================================
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Not Fraud', 'Fraud']
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
    
    print(f"\n{model_name} Performance:\n", classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    if y_pred_proba is not None:
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    plot_confusion_matrix(y_test, y_pred, model_name)


# ==========================================
# 8. Evaluate All Models
# ==========================================
for name, model in final_models.items():
    evaluate_model(model, X_test, y_test, name)


# ==========================================
# 9. Baseline (Existing System)
# ==========================================
y_pred_existing = fraud_flag.loc[X_test.index]

print("\nBaseline Fraud Flagging System Performance:")
print(classification_report(y_test, y_pred_existing))
plot_confusion_matrix(y_test, y_pred_existing, "Existing System")


# ==========================================
# 10. Feature Importance (for XGBoost)
# ==========================================
xgb_model = final_models["XGBoost"]
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n📌 Top Features for Fraud Detection (XGBoost):")
print(feature_importance.head(10))


# ==========================================
# 11. Save Best Model
# ==========================================
best_model = xgb_model   # picked after evaluation
joblib.dump(best_model, "fraud_model.pkl")
print("\n Model saved as fraud_model.pkl")
