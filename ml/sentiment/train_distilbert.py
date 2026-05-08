"""
Sentiment Analysis: Fine-tuned DistilBERT

Fine-tunes a pre-trained DistilBERT model on our product review data
for 3-class sentiment classification (Positive/Neutral/Negative).

Why DistilBERT?
- 40% smaller and 60% faster than BERT, with 97% of its performance
- Pre-trained on English text — already understands language structure
- Fine-tuning adapts this knowledge to our specific sentiment task
- Understands word context: "not good" = Negative (unlike TF-IDF)

Usage:
    python -m ml.sentiment.train_distilbert
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from flipkart.data_preprocessing import clean_text


# ── Configuration ──────────────────────────────────────────────
DATA_PATH = "data/flipkart_product_review.csv"
MODEL_DIR = "ml/sentiment/model/distilbert"
MODEL_NAME = "distilbert-base-uncased"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Training hyperparameters (tuned for small dataset + CPU)
NUM_EPOCHS = 3
BATCH_SIZE = 8          # Small batch for CPU memory
LEARNING_RATE = 2e-5    # Standard for fine-tuning transformers
MAX_LENGTH = 128        # Max token length (reviews are short)

LABEL_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}
LABEL_NAMES = ["Negative", "Neutral", "Positive"]


# ── Dataset Class ──────────────────────────────────────────────
class ReviewDataset(Dataset):
    """PyTorch Dataset for tokenized reviews.
    
    Converts raw text + labels into the format DistilBERT expects:
    - input_ids: tokenized text as integer IDs
    - attention_mask: 1 for real tokens, 0 for padding
    - labels: integer class labels
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=max_length,
            return_tensors=None  # Return lists, not tensors
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx]),
        }


# ── Helper Functions ───────────────────────────────────────────
def rating_to_sentiment(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"


def compute_class_weights(labels):
    """Compute class weights inversely proportional to frequency.
    
    Same idea as sklearn's class_weight='balanced'.
    Gives more importance to rare classes during training.
    """
    counts = np.bincount(labels, minlength=len(LABEL_NAMES))
    total = len(labels)
    weights = total / (len(LABEL_NAMES) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(eval_pred):
    """Custom metrics function for HuggingFace Trainer."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


# ── Custom Trainer with Class Weights ──────────────────────────
class WeightedTrainer(Trainer):
    """Custom Trainer that applies class weights to the loss function.
    
    Without this, the model would optimize for accuracy and just
    predict 'Positive' for everything (since it's 87% of the data).
    Class weights force the model to pay attention to rare classes.
    """
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Main Training Flow ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("Fine-tuning DistilBERT for Sentiment Analysis")
    print("=" * 60)
    
    # Load and prepare data
    print("\n[1/5] Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["review_clean"] = df["review"].astype(str).apply(clean_text)
    df = df[df["review_clean"].str.len() > 10]
    df["sentiment"] = df["rating"].apply(rating_to_sentiment)
    df["label"] = df["sentiment"].map(LABEL_MAP)
    
    X = df["review_clean"].tolist()
    y = df["label"].tolist()
    
    print(f"  Samples: {len(X)}")
    for name, idx in LABEL_MAP.items():
        count = y.count(idx)
        print(f"    {name}: {count} ({count/len(y)*100:.1f}%)")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Compute class weights
    class_weights = compute_class_weights(np.array(y_train))
    print(f"  Class weights: {dict(zip(LABEL_NAMES, class_weights.numpy().round(2)))}")
    
    # Tokenize
    print("\n[2/5] Tokenizing reviews...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = ReviewDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    test_dataset = ReviewDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    print(f"  Max token length: {MAX_LENGTH}")
    
    # Load model
    print("\n[3/5] Loading pre-trained DistilBERT...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_NAMES)
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="ml/sentiment/checkpoints",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=10,
        report_to="none",       # No W&B/MLflow for now
        use_cpu=True,           # Force CPU
    )
    
    # Train
    print(f"\n[4/5] Training ({NUM_EPOCHS} epochs, batch_size={BATCH_SIZE})...")
    print("  This will take ~15-20 minutes on CPU. Be patient!")
    start_time = time.time()
    
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    elapsed = time.time() - start_time
    print(f"\n  Training completed in {elapsed/60:.1f} minutes")
    
    # Evaluate
    print("\n[5/5] Evaluating...")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    
    print("\n" + "=" * 60)
    print("RESULTS: Fine-tuned DistilBERT")
    print("=" * 60)
    print(f"\n  Accuracy:          {accuracy:.4f}")
    print(f"  F1 (macro):        {f1_macro:.4f}")
    print(f"  F1 (weighted):     {f1_weighted:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))
    
    print(f"Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"{'':>12} {'Neg(pred)':>10} {'Neu(pred)':>10} {'Pos(pred)':>10}")
    for i, label in enumerate(LABEL_NAMES):
        print(f"  {label:>10} {cm[i][0]:>10} {cm[i][1]:>10} {cm[i][2]:>10}")
    
    # Save model
    print(f"\nSaving model to {MODEL_DIR}/...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    
    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "training_time_minutes": round(elapsed / 60, 1),
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "model_name": MODEL_NAME,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("  Done!")
    
    # Demo
    print("\n" + "=" * 60)
    print("DEMO: Try it on sample reviews")
    print("=" * 60)
    samples = [
        "Excellent sound quality, best purchase ever!",
        "Stopped working after 2 weeks, waste of money",
        "Decent product for the price, nothing special",
    ]
    model.eval()
    for text in samples:
        inputs = tokenizer(clean_text(text), return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = probs.argmax().item()
        print(f"  '{text}'")
        print(f"    -> {LABEL_NAMES[pred_idx]} (confidence: {probs[pred_idx]:.2f})")
        print()


if __name__ == "__main__":
    main()
