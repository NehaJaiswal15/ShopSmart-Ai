"""
Sentiment Analysis Inference Module

Loads the trained baseline model and provides a simple API
for predicting sentiment of product reviews.

Uses the TF-IDF + Logistic Regression baseline (not DistilBERT) because:
- Loads instantly (no GPU, no large model files)
- Prediction takes <1ms per review
- Good enough for real-time chatbot responses
- DistilBERT is kept for batch analysis / offline evaluation

Usage:
    from ml.sentiment.inference import SentimentPredictor

    predictor = SentimentPredictor()
    result = predictor.predict("Great sound quality!")
    # {'label': 'Positive', 'confidence': 0.85, 'emoji': '😊'}
"""

import os
import joblib
from flipkart.data_preprocessing import clean_text
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = "ml/sentiment/model"


class SentimentPredictor:
    """Lightweight sentiment predictor for real-time chatbot use."""

    # Emoji mapping for UI display
    EMOJI_MAP = {
        "Positive": "😊",
        "Neutral": "😐",
        "Negative": "😞",
    }

    def __init__(self, model_dir: str = MODEL_DIR):
        """Load the trained model and vectorizer from disk.

        Args:
            model_dir: Path to directory containing model files.

        Raises:
            FileNotFoundError: If model files don't exist (need to train first).
        """
        model_path = os.path.join(model_dir, "baseline_model.joblib")
        tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run 'python -m ml.sentiment.train_baseline' first."
            )

        self.model = joblib.load(model_path)
        self.tfidf = joblib.load(tfidf_path)
        logger.info("Sentiment model loaded successfully")

    def predict(self, text: str) -> dict:
        """Predict sentiment of a single review text.

        Args:
            text: Raw review text (will be cleaned automatically).

        Returns:
            dict with keys:
                - label: 'Positive', 'Neutral', or 'Negative'
                - confidence: float between 0 and 1
                - emoji: display emoji for the sentiment
        """
        cleaned = clean_text(text)
        features = self.tfidf.transform([cleaned])

        label = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = max(probabilities)

        return {
            "label": label,
            "confidence": round(confidence, 2),
            "emoji": self.EMOJI_MAP.get(label, ""),
        }

    def predict_batch(self, texts: list) -> list:
        """Predict sentiment for multiple reviews at once.

        Args:
            texts: List of raw review texts.

        Returns:
            List of prediction dicts.
        """
        cleaned = [clean_text(t) for t in texts]
        features = self.tfidf.transform(cleaned)

        labels = self.model.predict(features)
        probabilities = self.model.predict_proba(features)

        return [
            {
                "label": label,
                "confidence": round(max(proba), 2),
                "emoji": self.EMOJI_MAP.get(label, ""),
            }
            for label, proba in zip(labels, probabilities)
        ]
