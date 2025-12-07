# HIFS: Hybrid Imbalanced Fraud Scoring Framework for Credit Card Fraud Detection

## 📌 Overview
Credit card fraud transactions represent less than 0.2% of total transactions, making them extremely difficult to detect using traditional machine learning models.  
This project introduces **HIFS (Hybrid Imbalanced Fraud Scoring)** — a fraud detection framework combining:

- Isolation Forest (anomaly detection)
- SMOTE (oversampling)
- Soft Voting Ensemble (Logistic Regression + Random Forest + Gradient Boosting)

The framework significantly improves fraud recall and reduces false negatives, making it suitable for real-world banking deployments.

---

## 🗂 Project Structure
HIFS/
├─ src/main.py → model training & prediction
├─ data/creditcard.csv → dataset (not included in repo if restricted)
├─ outputs/ → generated model graphs
│ ├─ confusion_matrix.png
│ ├─ roc_curve.png
│ └─ feature_importance.png
├─ models/final_model.pkl → trained ensemble model
├─ requirements.txt → dependencies
└─ README.md → project documentation


---

## 📊 Results Summary
| Metric | Score |
|--------|--------|
| Accuracy | High |
| Precision | High |
| Recall | **Very High (primary focus)** |
| F1-Score | Strong |
| AUC | Excellent |

HIFS achieved the **best recall and F1-score** compared to individual models.

---

## 🧠 Key Features
✔ Handles extreme class imbalance  
✔ Extracts anomaly awareness using Isolation Forest  
✔ Uses ensemble predictions for higher model robustness  
✔ Deployable as an alert-based risk scoring engine  

---

## 🧩 Dataset Source
Kaggle — Credit Card Fraud Detection  
Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
*(Dataset not included due to file size policy — download manually if missing.)*

---

## 👤 Author
**Arnav Nigam**  
VIT Bhopal University  
2025

