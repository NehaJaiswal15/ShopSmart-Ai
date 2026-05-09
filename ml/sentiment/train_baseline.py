"""
Sentiment Analysis Baseline: TF-IDF + Logistic Regression

This script trains a simple but effective text classification model
as a baseline before attempting deep learning approaches.

Why start with a baseline?
- Trains in seconds (no GPU needed)
- Establishes a performance floor to beat
- Often surprisingly competitive on small datasets
- Shows interviewers you follow proper ML methodology

Usage:
    python -m ml.sentiment.train_baseline
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import joblib

from flipkart.data_preprocessing import clean_text


# ── Configuration ──────────────────────────────────────────────
DATA_PATH = "data/flipkart_product_review.csv"
MODEL_DIR = "ml/sentiment/model"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def rating_to_sentiment(rating: int) -> str:
    """Convert numeric rating to sentiment label.
    
    Mapping:
        1-2 → Negative
        3   → Neutral
        4-5 → Positive
    """
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"


def load_and_prepare_data(data_path: str):
    """Load CSV, clean text, create sentiment labels.
    
    Returns:
        X: cleaned review texts (list of strings)
        y: sentiment labels (list of strings)
    """
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Clean review text using our preprocessing pipeline
    df["review_clean"] = df["review"].astype(str).apply(clean_text)
    
    # Remove empty reviews after cleaning
    df = df[df["review_clean"].str.len() > 10]
    
    # Create sentiment labels from ratings
    df["sentiment"] = df["rating"].apply(rating_to_sentiment)
    
    print(f"  Total samples: {len(df)}")
    print(f"  Class distribution:")
    for label, count in df["sentiment"].value_counts().items():
        print(f"    {label}: {count} ({count/len(df)*100:.1f}%)")
    
    return df["review_clean"].tolist(), df["sentiment"].tolist()


def train_baseline(X, y):
    """Train TF-IDF + Logistic Regression pipeline.
    
    Key decisions:
    - class_weight='balanced': Automatically adjusts weights inversely
      proportional to class frequency. Critical for our imbalanced data
      (87% positive, 7% negative, 6% neutral).
    - TF-IDF max_features=5000: Limits vocabulary to top 5000 terms.
      Prevents overfitting on rare words in a small dataset.
    - ngram_range=(1,2): Uses both single words and 2-word phrases.
      Captures patterns like "not good", "very bad".
    """
    print("\nSplitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # Ensures proportional class representation in both sets
    )
    print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    
    # Step 1: Convert text to TF-IDF features
    print("\nTraining TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=5000,     # Top 5000 words only
        ngram_range=(1, 2),    # Unigrams + bigrams
        min_df=2,              # Ignore words appearing in < 2 documents
        max_df=0.95,           # Ignore words appearing in > 95% of documents
        sublinear_tf=True,     # Apply log normalization to term frequencies
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"  Vocabulary size: {len(tfidf.vocabulary_)} features")
    
    # Step 2: Train Logistic Regression
    print("\nTraining Logistic Regression (with class weights)...")
    model = LogisticRegression(
        class_weight="balanced",  # Handle class imbalance
        max_iter=1000,
        random_state=RANDOM_STATE,
        C=1.0,                    # Regularization strength
    )
    model.fit(X_train_tfidf, y_train)
    
    # Step 3: Evaluate
    print("\n" + "=" * 60)
    print("RESULTS: TF-IDF + Logistic Regression Baseline")
    print("=" * 60)
    
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    
    print(f"\n  Accuracy:          {accuracy:.4f}")
    print(f"  F1 (macro):        {f1_macro:.4f}")
    print(f"  F1 (weighted):     {f1_weighted:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"Confusion Matrix:")
    labels = ["Negative", "Neutral", "Positive"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    # Pretty print the confusion matrix
    print(f"{'':>12} {'Neg(pred)':>10} {'Neu(pred)':>10} {'Pos(pred)':>10}")
    for i, label in enumerate(labels):
        print(f"  {label:>10} {cm[i][0]:>10} {cm[i][1]:>10} {cm[i][2]:>10}")
    
    # Step 4: Cross-validation (more robust estimate)
    print(f"\n5-Fold Cross-Validation:")
    X_all_tfidf = tfidf.transform(X)
    cv_scores = cross_val_score(model, X_all_tfidf, y, cv=5, scoring="f1_macro")
    print(f"  F1 (macro) scores: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return model, tfidf, {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "cv_f1_mean": round(cv_scores.mean(), 4),
        "cv_f1_std": round(cv_scores.std(), 4),
    }


def save_model(model, tfidf, metrics):
    """Save trained model, vectorizer, and metrics to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    joblib.dump(model, os.path.join(MODEL_DIR, "baseline_model.joblib"))
    joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    
    with open(os.path.join(MODEL_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nModel saved to {MODEL_DIR}/")
    print(f"  - baseline_model.joblib")
    print(f"  - tfidf_vectorizer.joblib")
    print(f"  - baseline_metrics.json")


if __name__ == "__main__":
    X, y = load_and_prepare_data(DATA_PATH)
    model, tfidf, metrics = train_baseline(X, y)
    save_model(model, tfidf, metrics)
    
    # Quick demo
    print("\n" + "=" * 60)
    print("DEMO: Try it on sample reviews")
    print("=" * 60)
    samples = [
        "Excellent sound quality, best purchase ever!",
        "Stopped working after 2 weeks, waste of money",
        "Decent product for the price, nothing special",
    ]
    for text in samples:
        clean = clean_text(text)
        features = tfidf.transform([clean])
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        confidence = max(proba)
        print(f"  '{text}'")
        print(f"    -> {pred} (confidence: {confidence:.2f})")
        print()
