"""
train_model.py
Trains the Random Forest Classifier on the health risk dataset and saves the model,
scaler, and feature columns for use by the Flask application.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ──────────────────────────────────────────────
# 1. Load dataset
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Lifestyle_and_Health_Risk_Prediction_Synthetic_Dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ──────────────────────────────────────────────
# 2. Drop missing target rows
# ──────────────────────────────────────────────
df = df.dropna(subset=["health_risk"])

# ──────────────────────────────────────────────
# 3. Impute missing values
# ──────────────────────────────────────────────
df["age"]    = df["age"].fillna(df["age"].mean())
df["weight"] = df["weight"].fillna(df["weight"].mean())
df["height"] = df["height"].fillna(df["height"].median())
df["sleep"]  = df["sleep"].fillna(df["sleep"].median())
df["bmi"]    = df["bmi"].fillna(df["bmi"].mean())

# ──────────────────────────────────────────────
# 4. Label-encode binary columns
# ──────────────────────────────────────────────
le_smoking = LabelEncoder()
le_alcohol = LabelEncoder()
le_married = LabelEncoder()

df["smoking_encoded"] = le_smoking.fit_transform(df["smoking"])
df["alcohol_encoded"] = le_alcohol.fit_transform(df["alcohol"])
df["married_encoded"] = le_married.fit_transform(df["married"])

# Save label encoder mappings
smoking_map = dict(zip(le_smoking.classes_, le_smoking.transform(le_smoking.classes_).tolist()))
alcohol_map = dict(zip(le_alcohol.classes_, le_alcohol.transform(le_alcohol.classes_).tolist()))
married_map = dict(zip(le_married.classes_, le_married.transform(le_married.classes_).tolist()))

# ──────────────────────────────────────────────
# 5. One-hot encode exercise & profession
# ──────────────────────────────────────────────
exercise_dummies  = pd.get_dummies(df["exercise"],  prefix="exercise")
profession_dummies = pd.get_dummies(df["profession"], prefix="profession")
df = pd.concat([df, exercise_dummies, profession_dummies], axis=1)

# ──────────────────────────────────────────────
# 6. Age category & engineered feature
# ──────────────────────────────────────────────
bins   = [0, 30, 45, 60, df["age"].max() + 1]
labels = ["Young", "Middle-Aged", "Senior", "Elderly"]
df["age_category"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True)

df["bmi_sleep_interaction"] = df["bmi"] * df["sleep"]

# ──────────────────────────────────────────────
# 7. Build feature matrix (same order as notebook)
# ──────────────────────────────────────────────
drop_cols = ["smoking", "alcohol", "married", "exercise", "profession",
             "sugar_intake", "age_category", "health_risk"]
# Drop Cluster if it exists (it won't at this point, but guard anyway)
if "Cluster" in df.columns:
    drop_cols.append("Cluster")

df_ml = df.drop(columns=drop_cols)
X = df_ml.copy()

# Encode target
le_risk = LabelEncoder()
y = le_risk.fit_transform(df["health_risk"])
print(f"Target mapping: {dict(zip(le_risk.classes_, le_risk.transform(le_risk.classes_).tolist()))}")

# ──────────────────────────────────────────────
# 8. Train / test split & scale
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ──────────────────────────────────────────────
# 9. Train Random Forest (same params as notebook)
# ──────────────────────────────────────────────
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf_model.fit(X_train_scaled, y_train)

y_pred = rf_model.predict(X_test_scaled)
acc    = accuracy_score(y_test, y_pred)
print(f"\nRandom Forest Test Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_risk.classes_))

# ──────────────────────────────────────────────
# 10. Save artifacts
# ──────────────────────────────────────────────
joblib.dump(rf_model, os.path.join(MODEL_DIR, "rf_model.pkl"))
joblib.dump(scaler,   os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_columns.pkl"))
joblib.dump(le_risk,  os.path.join(MODEL_DIR, "le_risk.pkl"))

# Save category mappings for the front-end
mappings = {
    "smoking":    smoking_map,
    "alcohol":    alcohol_map,
    "married":    married_map,
    "exercise_levels":  sorted(df["exercise"].unique().tolist()),
    "profession_list":  sorted(df["profession"].unique().tolist()),
    "exercise_columns":  sorted([c for c in X.columns if c.startswith("exercise_")]),
    "profession_columns": sorted([c for c in X.columns if c.startswith("profession_")]),
}
joblib.dump(mappings, os.path.join(MODEL_DIR, "mappings.pkl"))

print("\nAll artifacts saved to /model/")
print(f"Feature columns ({len(X.columns)}): {list(X.columns)}")
