# 🛒 ShopSmart AI

An AI-powered e-commerce chatbot that combines **Retrieval-Augmented Generation (RAG)** with **real-time sentiment analysis** to help users explore, compare, and get recommendations on products.

## Architecture

```
User (Browser)
     │
     ▼
┌──────────────────────────────────────────┐
│            Flask Web Server              │
│                                          │
│  ┌──────────────────────────────────┐    │
│  │       RAG Chain (LangChain)      │    │
│  │                                  │    │
│  │  ┌──────────┐  ┌─────────────┐  │    │
│  │  │ AstraDB  │  │  Groq LLM   │  │    │
│  │  │ (Vector  │  │ (LLaMA 3.1) │  │    │
│  │  │  Store)  │  │             │  │    │
│  │  └──────────┘  └─────────────┘  │    │
│  └──────────────────────────────────┘    │
│                                          │
│  ┌──────────────────────────────────┐    │
│  │     Sentiment Predictor          │    │
│  │  TF-IDF + Logistic Regression    │    │
│  └──────────────────────────────────┘    │
│                                          │
│  ┌──────────────────────────────────┐    │
│  │      Prometheus Metrics          │    │
│  └──────────────────────────────────┘    │
└──────────────────────────────────────────┘
```

## Features

- **Conversational AI** — Natural language product Q&A with chat history awareness
- **RAG Pipeline** — Retrieves semantically relevant product reviews from AstraDB vector store using HuggingFace BGE embeddings
- **Anti-Hallucination** — System prompt constrains the LLM to only use provided context; never invents prices or ratings
- **Sentiment Analysis** — Real-time ML-based sentiment classification of retrieved reviews (TF-IDF baseline + DistilBERT)
- **Data Pipeline** — Automated preprocessing with schema validation, text cleaning, and quality filtering
- **Monitoring** — Prometheus-compatible `/metrics` endpoint for production observability

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Python, Flask |
| **LLM** | Groq API (LLaMA 3.1-8B) |
| **RAG Framework** | LangChain |
| **Vector Store** | AstraDB (Cassandra) |
| **Embeddings** | HuggingFace BGE |
| **ML - Baseline** | TF-IDF + Logistic Regression (scikit-learn) |
| **ML - Deep Learning** | DistilBERT (HuggingFace Transformers) |
| **Monitoring** | Prometheus |
| **Frontend** | HTML, CSS, JavaScript, Bootstrap |

## Project Structure

```
ShopSmart-Ai/
├── app.py                          # Flask application entry point
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── Dockerfile                      # Container configuration
│
├── flipkart/                       # Core RAG pipeline
│   ├── config.py                   # Configuration (env vars)
│   ├── data_converter.py           # CSV → LangChain Document conversion
│   ├── data_ingestion.py           # AstraDB vector store ingestion
│   ├── data_preprocessing.py       # Text cleaning & validation
│   └── rag_chain.py                # LangChain RAG chain builder
│
├── ml/                             # Machine Learning pipeline
│   └── sentiment/
│       ├── train_baseline.py       # TF-IDF + LogReg training
│       ├── train_distilbert.py     # DistilBERT fine-tuning
│       ├── inference.py            # Real-time prediction API
│       └── model/                  # Saved model artifacts (git-ignored)
│
├── notebooks/                      # Analysis & evaluation
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   └── 02_model_comparison.ipynb   # Baseline vs DistilBERT comparison
│
├── data/                           # Dataset
│   ├── flipkart_product_review.csv # Raw product reviews
│   └── product_prices.csv          # Product pricing data
│
├── utils/                          # Shared utilities
│   ├── logger.py                   # Structured logging
│   └── custom_exception.py         # Custom exception handling
│
├── templates/                      # Frontend
│   └── index.html                  # Chat UI
├── static/
│   └── style.css                   # Chat styling
│
└── prometheus/                     # Monitoring config
```

## Setup & Installation

### Prerequisites
- Python 3.10+
- [AstraDB account](https://astra.datastax.com/) (free tier)
- [Groq API key](https://console.groq.com/) (free tier)
- [HuggingFace API token](https://huggingface.co/settings/tokens)

### 1. Clone & Install

```bash
git clone https://github.com/NehaJaiswal15/ShopSmart-Ai.git
cd ShopSmart-Ai

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required variables (see `.env.example`):
- `GROQ_API_KEY`
- `HF_TOKEN`
- `ASTRA_DB_API_ENDPOINT`
- `ASTRA_DB_APPLICATION_TOKEN`
- `ASTRA_DB_KEYSPACE`

### 3. Train Sentiment Model

```bash
# Train baseline (2 seconds)
python -m ml.sentiment.train_baseline

# Train DistilBERT (15-20 min on CPU)
python -m ml.sentiment.train_distilbert
```

### 4. Run the Application

```bash
python app.py
# Open http://localhost:5000
```

## ML Pipeline

### Sentiment Analysis

Two models trained on 443 product reviews with 3-class classification (Positive/Neutral/Negative):

| Model | Accuracy | F1 (macro) | Training Time | Inference |
|-------|----------|-----------|--------------|-----------|
| TF-IDF + LogReg | 88.8% | 0.47 | ~2 sec | <1ms |
| **DistilBERT** | **93.3%** | **0.63** | ~6.5 min | ~50ms |

**Production choice:** TF-IDF baseline is used for real-time chatbot enrichment (speed). DistilBERT is available for batch analysis (accuracy).

**Class imbalance handling:** 87% Positive / 7% Negative / 6% Neutral — addressed with stratified splits, balanced class weights, and custom `WeightedTrainer`.

### Data Pipeline

1. **Schema validation** — Rejects malformed data before processing
2. **Text cleaning** — HTML stripping, URL removal, whitespace normalization
3. **Quality filtering** — Removes reviews under 10 characters
4. **Price enrichment** — Merges product pricing into review documents
5. **Vector ingestion** — Embeds and stores in AstraDB

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat UI |
| `/get` | POST | Send message, get AI response |
| `/sentiment` | POST | Standalone sentiment analysis |
| `/metrics` | GET | Prometheus metrics |

### Sentiment API Example

```bash
curl -X POST http://localhost:5000/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "Great sound quality, love this product!"}'

# Response: {"label": "Positive", "confidence": 0.85, "emoji": "😊"}
```

## License

This project is for educational and portfolio purposes.
