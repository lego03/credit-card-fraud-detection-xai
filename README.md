# Credit Card Fraud Detection using Ensemble Learning with Explainable AI

## Problem Statement
Detecting credit card fraud using machine learning, with a focus on how different class-imbalance handling techniques affect both predictive performance and the reliability of model explanations (SHAP).

## Motivation
Credit card fraud detection is a highly imbalanced classification problem — fraudulent transactions make up less than 0.2% of all transactions. Most existing studies apply imbalance-handling techniques (like SMOTE) and explainability tools (like SHAP) independently, without checking whether the choice of imbalance-handling method affects the *reliability and consistency* of the resulting explanations. This project investigates that gap directly.

## Dataset
- **Credit Card Fraud Detection dataset** (Kaggle, ULB Machine Learning Group)
- 284,807 transactions, 492 labeled frauds (0.173%)
- 30 features: Time, Amount, and 28 PCA-transformed features (V1–V28)
- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Methodology
1. **Preprocessing**: Stratified 80/20 train-test split; Amount and Time scaled using StandardScaler (fit on training data only, to avoid data leakage)
2. **Models compared**: Logistic Regression, Random Forest, XGBoost
3. **Imbalance-handling strategies compared**: No resampling (baseline), SMOTE, Class-Weighting
4. **Evaluation metrics**: Precision, Recall, F1-score (not accuracy alone, since accuracy is misleading on this heavily imbalanced dataset)
5. **Explainability**: SHAP (TreeExplainer) applied to Random Forest and XGBoost under both SMOTE and Class-Weighting, comparing feature importance rankings using Spearman rank correlation

## Results

### Predictive Performance (fraud class only)

| Model | Imbalance Method | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | None | 0.84 | 0.69 | 0.76 |
| Random Forest | None | 0.94 | 0.82 | 0.87 |
| Random Forest | SMOTE | 0.85 | 0.84 | 0.84 |
| Random Forest | Class-Weight | 0.96 | 0.74 | 0.84 |
| XGBoost | SMOTE | 0.24 | 0.89 | 0.38 |
| XGBoost | Class-Weight | 0.50 | 0.85 | 0.63 |

### SHAP Explanation Stability (SMOTE vs Class-Weight)

| Model | Spearman Correlation |
|---|---|
| Random Forest | 0.98 |
| XGBoost | 0.69 |

## Key Finding
Random Forest's SHAP explanations remain highly consistent regardless of which imbalance-handling technique is used (Spearman ρ = 0.98), while XGBoost's explanations shift meaningfully depending on the method chosen (ρ = 0.69) — despite XGBoost also showing much larger swings in predictive performance (F1 = 0.38 with SMOTE vs 0.63 with Class-Weight) across the same two methods. This suggests that **model architecture, not just the imbalance-handling technique itself, determines how trustworthy and reproducible SHAP explanations are** — an interaction not addressed in prior fraud detection literature reviewed for this project.

## How to Run
1. Open `fraud_detection.ipynb` in Google Colab
2. Set up a Kaggle API token 
3. Run all cells in order — the notebook downloads the dataset, trains all models, and generates all evaluation metrics and SHAP visualizations

## Tools & Libraries
Python, scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP, pandas, matplotlib

## Future Work
- Validate findings on a second public dataset (e.g., IEEE-CIS Fraud Detection) to test generalization
- Extend explanation-stability comparison to LIME in addition to SHAP
- Deploy as an interactive demo (e.g., Streamlit) for real-time predictions with live explanations
