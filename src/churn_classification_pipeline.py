"""
Predictive Outcome & Customer Trend Modeling
Supervised Classification Pipeline on 25,000+ Customer Records
Author: Rahul
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score


def generate_customer_data(n_samples: int = 25000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    tenure_months = rng.integers(1, 60, size=n_samples)
    monthly_spend = rng.normal(1450, 420, size=n_samples).clip(200, 5000)
    support_tickets = rng.poisson(1.4, size=n_samples)
    days_since_last_login = rng.exponential(18, size=n_samples).astype(int)
    contract_type = rng.choice(["Month-to-Month", "Quarterly", "Annual"], size=n_samples, p=[0.55, 0.25, 0.20])

    logit = (
        -1.6
        - 0.04 * tenure_months
        + 0.045 * days_since_last_login
        + 0.38 * support_tickets
        + np.where(contract_type == "Month-to-Month", 0.85, -0.4)
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob)

    return pd.DataFrame({
        "tenure_months": tenure_months,
        "monthly_spend": np.round(monthly_spend, 2),
        "support_tickets": support_tickets,
        "days_since_last_login": days_since_last_login,
        "contract_type": contract_type,
        "churn": churn
    })


def run_pipeline():
    df = generate_customer_data(25000)
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_features = ["tenure_months", "monthly_spend", "support_tickets", "days_since_last_login"]
    categorical_features = ["contract_type"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features)
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= 0.58).astype(int)

    print("=== Customer Churn Classification Report ===")
    print(classification_report(y_test, y_pred))
    print(f"Precision: {precision_score(y_test, y_pred):.2%}")
    print(f"Recall:    {recall_score(y_test, y_pred):.2%}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_probs):.4f}")


if __name__ == "__main__":
    run_pipeline()
