# Credit Card Fraud Detection with Imbalanced Data

A Machine Learning project that detects potentially fraudulent credit card transactions using Logistic Regression, class weighting, SMOTE, and probability threshold tuning.

## Project Overview

Credit card fraud detection is a challenging classification problem because fraudulent transactions represent only a very small proportion of all transactions.

This project explores how class imbalance affects model performance and evaluates different strategies for identifying fraudulent transactions while controlling false alarms.

## Objectives

- Explore and analyze an imbalanced transaction dataset.
- Perform exploratory data analysis (EDA).
- Build a Logistic Regression baseline model.
- Compare class weighting and SMOTE.
- Evaluate models using fraud-focused metrics.
- Tune the classification threshold using a validation set.
- Save the trained model and preprocessing artifacts.

## Dataset

**Dataset:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains 284,807 transactions and 30 input features.

- `Class = 0`: Legitimate transaction
- `Class = 1`: Fraudulent transaction

There are 492 fraudulent transactions, approximately 0.173% of the complete dataset.

The `V1`–`V28` features are anonymized numerical features produced through PCA transformation. `Time` and `Amount` are also included.

Download the dataset from Kaggle and place `creditcard.csv` in the appropriate local data directory before running the notebook.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- imbalanced-learn (SMOTE)
- Joblib
- Jupyter Notebook

## Workflow

1. Data loading and inspection
2. Exploratory Data Analysis
3. Missing-value and duplicate checks
4. Feature-target separation
5. Stratified train/test split
6. Feature scaling
7. Baseline Logistic Regression
8. Class Weight and SMOTE experiments
9. Precision, Recall, F1-score, ROC-AUC, and PR-AUC evaluation
10. Validation-based threshold selection
11. Final test-set evaluation
12. Model serialization

## Handling Class Imbalance

Three approaches were explored:

- **Baseline Logistic Regression:** Trained on the original imbalanced training data.
- **Class Weight:** Used `class_weight="balanced"` to penalize errors on the minority class more heavily.
- **SMOTE:** Generated synthetic minority-class examples on the training data only.

The test set was kept at its original class distribution.

## Final Model Results

The selected Logistic Regression model uses a probability threshold of 0.1, selected through validation experiments.

| Metric | Test Result |
|---|---:|
| Fraud Precision | 83.33% |
| Fraud Recall | 73.68% |
| Fraud F1-score | 78.21% |
| ROC-AUC | 95.60% |
| PR-AUC (Average Precision) | 69.20% |
| True Positives | 70 |
| False Positives | 14 |
| True Negatives | 56,637 |
| False Negatives | 25 |

These results indicate that the model identified 70 of the 95 fraudulent transactions in the test set, while missing 25 and incorrectly flagging 14 legitimate transactions.

Performance should be interpreted using precision, recall, and PR-AUC rather than accuracy alone because the dataset is highly imbalanced.

## Model Artifacts

The trained model, fitted scaler, feature names, and selected threshold can be saved together using Joblib.

The artifact is intended for local reuse or deployment. Do not load serialized model files from untrusted sources.

## How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd fraud-detection-imbalanced-data
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `creditcard.csv` from the Kaggle dataset page and place it where the notebook expects it.

### 5. Run the notebook

```bash
jupyter notebook fraud_detection.ipynb
```

Run the notebook cells in order.

## Limitations

- The `V1`–`V28` features are anonymized and cannot be directly interpreted as real-world transaction attributes.
- Threshold selection depends on the relative costs of missed fraud and false alarms.
- SMOTE does not guarantee better generalization.
- Results from a single random split may not represent future real-world transaction performance.
- This project is educational and is not validated for production financial decisions.

## Future Improvements

- Compare Random Forest and gradient boosting models.
- Tune thresholds using business costs and validation data.
- Evaluate calibration and precision-recall curves.
- Use cross-validation and test for temporal distribution shifts.
- Build a web interface for transaction risk predictions.
- Add monitoring for changes in fraud patterns.

## Author

**Ali Muhammad**

Machine Learning | AI & Data Science

GitHub: [alimuhammad-007](https://github.com/alimuhammad-007)

---

*This project was developed for learning and portfolio demonstration purposes.*
