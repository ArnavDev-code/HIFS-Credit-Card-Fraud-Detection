# src/main.py
"""
HIFS - Hybrid Imbalanced Fraud Scoring
Run: python src/main.py

This script:
- Loads creditcard.csv (place in ../data/)
- Performs quick EDA and saves plots
- Adds IsolationForest anomaly score as a feature
- Uses SMOTE to resample training data
- Trains a VotingClassifier (LR + RF + GradientBoosting)
- Evaluates on test set and saves visual outputs and model
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import joblib

# ---------- Paths ----------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "creditcard.csv")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- Load dataset ----------
print("Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Class distribution:\n", df['Class'].value_counts())

# ---------- Quick EDA ----------
def quick_eda(df):
    print("\nRunning quick EDA...")
    # Save class distribution plot
    plt.figure(figsize=(6,4))
    sns.countplot(x='Class', data=df)
    plt.title("Class Distribution (0 = legit, 1 = fraud)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))
    plt.clf()

    # Correlation heatmap on a sample (to speed up)
    sample = df.sample(n=5000, random_state=42) if df.shape[0] > 5000 else df.copy()
    corr = sample.corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap='coolwarm', vmax=0.8)
    plt.title("Correlation Heatmap (sample)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
    plt.clf()
    print("EDA plots saved to outputs/")

quick_eda(df)

# ---------- Preprocessing ----------
# Separate features and target
X = df.drop(columns=['Class'])
y = df['Class'].copy()

# Scale Time and Amount (important)
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------- IsolationForest anomaly scores ----------
# Fit on training data only
iso = IsolationForest(contamination=y.mean(), random_state=42)
iso.fit(X_train)
train_iso_scores = -iso.score_samples(X_train)  # higher means more anomalous
test_iso_scores = -iso.score_samples(X_test)

# Append as new column
X_train = X_train.copy()
X_test = X_test.copy()
X_train['iso_score'] = train_iso_scores
X_test['iso_score'] = test_iso_scores

# ---------- Handle imbalance with SMOTE ----------
print("\nBefore SMOTE, value counts:\n", y_train.value_counts())
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
print("After SMOTE, value counts:\n", pd.Series(y_res).value_counts())

# ---------- Build models ----------
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
gb = GradientBoostingClassifier(random_state=42)

voting = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('gb', gb)],
    voting='soft',
    n_jobs=-1
)

# ---------- Train ----------
print("\nTraining VotingClassifier on resampled data...")
voting.fit(X_res, y_res)
joblib.dump(voting, os.path.join(MODEL_DIR, "hifs_voting_model.pkl"))
print("Model saved to models/hifs_voting_model.pkl")

# ---------- Evaluate ----------
print("\nEvaluating on test set...")
y_pred = voting.predict(X_test)
y_proba = voting.predict_proba(X_test)[:, 1]

print("\nClassification report:\n")
print(classification_report(y_test, y_pred, digits=4))

auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC:", auc)

# Save classification report CSV
report = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(report).T.to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"))
print("Saved classification_report.csv")

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.clf()
print("Saved confusion_matrix.png")

# ROC curve plot
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], '--', linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUC = {auc:.4f})")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
plt.clf()
print("Saved roc_curve.png")

# Feature importance using RandomForest inside VotingClassifier
rf_imp = voting.named_estimators_['rf'].feature_importances_
feat_names = X_test.columns
imp_df = pd.DataFrame({'feature': feat_names, 'importance': rf_imp})
imp_df = imp_df.sort_values(by='importance', ascending=False).head(20)

plt.figure(figsize=(8,6))
sns.barplot(x='importance', y='feature', data=imp_df)
plt.title("Top 20 Feature Importances (from RF)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
plt.clf()
print("Saved feature_importance.png")

print("\nAll done. Check the outputs/ folder for visuals and models/.")
