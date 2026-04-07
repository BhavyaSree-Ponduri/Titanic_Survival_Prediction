"""
model.py - ML Model Training, Evaluation, and Persistence
Titanic Survival Prediction System
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.impute import SimpleImputer

MODEL_PATH = "titanic_model.pkl"
SCALER_PATH = "titanic_scaler.pkl"
DATA_PATH = "dataset.csv"

FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']


def load_and_preprocess(path: str = DATA_PATH):
    """Load CSV and return cleaned feature matrix X and target y."""
    df = pd.read_csv(path)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Drop rows where Age or Fare is still missing after imputation setup
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')

    imputer = SimpleImputer(strategy='median')
    df[['Age', 'Fare']] = imputer.fit_transform(df[['Age', 'Fare']])

    df = df.dropna(subset=['Survived', 'Pclass', 'Sex'])

    X = df[FEATURES].values
    y = df['Survived'].values.astype(int)
    return X, y, df


def train_all_models(path: str = DATA_PATH):
    """Train LR, DT, RF; return metrics dict and best model name."""
    X, y, _ = load_and_preprocess(path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    best_model_name = None
    best_acc = 0

    for name, model in models.items():
        X_tr = X_train_sc if name == "Logistic Regression" else X_train
        X_te = X_test_sc if name == "Logistic Regression" else X_test
        model.fit(X_tr, y_train)
        preds = model.predict(X_te)
        acc = accuracy_score(y_test, preds)
        results[name] = {
            "model": model,
            "accuracy": round(acc * 100, 2),
            "precision": round(precision_score(y_test, preds, zero_division=0) * 100, 2),
            "recall": round(recall_score(y_test, preds, zero_division=0) * 100, 2),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
            "predictions": preds.tolist(),
            "y_test": y_test.tolist(),
        }
        if acc > best_acc:
            best_acc = acc
            best_model_name = name

    # Save best model
    best = results[best_model_name]["model"]
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    return results, best_model_name, scaler


def load_model():
    """Load saved model and scaler (train if not found)."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        train_all_models()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def predict_single(pclass: int, sex: str, age: float, sibsp: int,
                   parch: int, fare: float, model=None, scaler=None):
    """Return (survived: bool, probability: float)."""
    if model is None or scaler is None:
        model, scaler = load_model()
    sex_enc = 1 if sex.lower() == "female" else 0
    X = np.array([[pclass, sex_enc, age, sibsp, parch, fare]])
    prob = model.predict_proba(X)[0][1]
    survived = prob >= 0.5
    return survived, round(prob * 100, 2)


def predict_batch(df: pd.DataFrame, model=None, scaler=None):
    """Predict survival for a DataFrame; return df with Survived and Probability columns."""
    if model is None or scaler is None:
        model, scaler = load_model()
    df = df.copy()
    df['Sex'] = df['Sex'].str.lower().map({'male': 0, 'female': 1}).fillna(0)
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(df['Age'].median() if 'Age' in df else 29)
    df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce').fillna(14)
    for col in ['Pclass', 'SibSp', 'Parch']:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    X = df[FEATURES].values
    probs = model.predict_proba(X)[:, 1]
    df['Survived'] = (probs >= 0.5).astype(int)
    df['Survival Probability (%)'] = (probs * 100).round(2)
    return df


def get_feature_importance(model=None):
    """Return feature importances as dict (RF/DT only)."""
    if model is None:
        model, _ = load_model()
    if hasattr(model, 'feature_importances_'):
        return dict(zip(FEATURES, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        return dict(zip(FEATURES, abs(model.coef_[0])))
    return {}
