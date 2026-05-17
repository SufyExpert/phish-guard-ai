"""
phishing_train_model.py
=======================
Trains a Logistic Regression classifier on the Phishing_Email.csv dataset.
Pipeline: Raw Email Text → Preprocessing → TF-IDF Vectorization → Logistic Regression

Dataset columns:
  (index)  |  Email Text  |  Email Type  ('Safe Email' or 'Phishing Email')
"""

import os
import re
import string
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
from sklearn.pipeline import Pipeline

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE_DIR, "Phishing_Email.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)


# ── 1. Load Dataset ────────────────────────────────────────────────────────────
print("=" * 60)
print(" PHISHING EMAIL DETECTION — MODEL TRAINING")
print("=" * 60)

df = pd.read_csv(CSV_PATH, index_col=0)
print(f"\n[1] Raw dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"    Columns: {list(df.columns)}")


# ── 2. Rename & Drop Garbage Rows ──────────────────────────────────────────────
df.columns = ["text", "label"]
print(f"\n[2] Label distribution (raw):\n{df['label'].value_counts()}")

# Drop rows where text or label is null / blank / whitespace-only
before = len(df)
df = df.dropna(subset=["text", "label"])
df["text"]  = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip()
df = df[df["text"].str.len() > 5]          # discard near-empty cells
df = df[df["label"].isin(["Safe Email", "Phishing Email"])]
after = len(df)
print(f"\n[3] After cleaning: {after:,} rows kept ({before - after:,} dropped)")
print(f"    Label distribution:\n{df['label'].value_counts()}")


# ── 3. Text Preprocessing ──────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Lightweight but effective cleaning:
    - lowercase
    - strip URLs
    - strip email addresses
    - strip HTML tags
    - remove punctuation & digits (keep alphabetical tokens)
    - collapse whitespace
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)          # URLs → token
    text = re.sub(r"\S+@\S+", " email ", text)                  # addresses → token
    text = re.sub(r"<[^>]+>", " ", text)                        # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                       # strip non-alpha
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("\n[4] Preprocessing email text …")
df["clean_text"] = df["text"].apply(preprocess_text)

# Remove entries that collapse to empty strings after preprocessing
df = df[df["clean_text"].str.len() > 0]
print(f"    Done. {len(df):,} usable samples remain.")


# ── 4. Encode Labels ───────────────────────────────────────────────────────────
label_map = {"Safe Email": 0, "Phishing Email": 1}
df["y"] = df["label"].map(label_map)


# ── 5. Train / Test Split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["y"],
    test_size=0.2, random_state=42, stratify=df["y"]
)
print(f"\n[5] Train: {len(X_train):,} | Test: {len(X_test):,}")


# ── 6. Build Pipeline (TF-IDF + Logistic Regression) ──────────────────────────
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=40_000,
        ngram_range=(1, 2),        # unigrams + bigrams
        sublinear_tf=True,         # log-scaled TF
        min_df=3,                  # ignore very rare terms
        max_df=0.90,               # ignore very common terms
        strip_accents="unicode",
        analyzer="word",
    )),
    ("clf", LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",   # handles any class imbalance
        random_state=42,
    )),
])


# ── 7. Fit ─────────────────────────────────────────────────────────────────────
print("\n[6] Training TF-IDF + Logistic Regression …")
pipeline.fit(X_train, y_train)


# ── 8. Evaluate ────────────────────────────────────────────────────────────────
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
auc     = roc_auc_score(y_test, y_proba)
cm      = confusion_matrix(y_test, y_pred)

print(f"\n[7] Test Accuracy : {acc * 100:.2f}%")
print(f"    ROC-AUC Score  : {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Safe Email", "Phishing Email"]))
print("Confusion Matrix:")
print(f"  True Safe (TN) : {cm[0,0]:>6,}    False Phish (FP) : {cm[0,1]:>6,}")
print(f"  False Safe (FN): {cm[1,0]:>6,}    True Phish  (TP) : {cm[1,1]:>6,}")


# ── 9. Save Artifacts ─────────────────────────────────────────────────────────
joblib.dump(pipeline, os.path.join(MODEL_DIR, "phishing_pipeline.pkl"))

metadata = {
    "label_map":      label_map,
    "inv_label_map":  {v: k for k, v in label_map.items()},
    "accuracy":       round(acc * 100, 2),
    "roc_auc":        round(auc, 4),
    "train_samples":  int(len(X_train)),
    "test_samples":   int(len(X_test)),
    "total_samples":  int(len(df)),
    "vocab_size":     int(pipeline.named_steps["tfidf"].max_features),
    "ngram_range":    "(1, 2)",
    "model":          "Logistic Regression",
    "vectorizer":     "TF-IDF",
}
joblib.dump(metadata, os.path.join(MODEL_DIR, "metadata.pkl"))

print("\n[8] Artifacts saved to /model/")
print("    ✓  phishing_pipeline.pkl")
print("    ✓  metadata.pkl")
print("\nTraining complete!")
