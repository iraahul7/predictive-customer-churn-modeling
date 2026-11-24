# Predictive Outcome & Customer Trend Modeling

**Tech Stack:** Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn) | Classification Modeling | Feature Engineering

## Project Overview
Built an end-to-end supervised classification pipeline (EDA, missing-value imputation, categorical encoding, hyperparameter validation) on **25,000+ customer records** to flag high-risk churn segments, achieving **82% precision and 78% recall**.

![Churn Model Confusion Matrix and ROC Curve](images/churn_model_evaluation.png)

## Repository Structure
```text
|-- notebooks/
|   |-- 01_churn_modeling_and_evaluation.ipynb  # Executed notebook with classification report & ROC curve
|-- src/
|   |-- churn_classification_pipeline.py        # Complete ML pipeline: preprocessing, Random Forest & evaluation
|-- images/
|   |-- churn_model_evaluation.png              # Confusion Matrix & ROC-AUC visual plot
|-- requirements.txt
|-- README.md
```

## Model Performance & Business Strategy
- **Precision:** 82% (minimizing wasted retention discounts on low-risk users)
- **Recall:** 78% (capturing nearly 4 out of 5 at-risk customers prior to churn)
- **Actionable Retention Playbook:** Segmented flagged customers by tenure and support ticket frequency to trigger targeted outreach campaigns.
