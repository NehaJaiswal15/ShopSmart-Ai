from flask import render_template, Flask, request, Response, jsonify
from prometheus_client import Counter, generate_latest

from flipkart.data_ingestion import DataIngestor
from flipkart.rag_chain import RAGChainBuilder
from ml.sentiment.inference import SentimentPredictor
from utils.logger import get_logger

from dotenv import load_dotenv
load_dotenv()

logger = get_logger(__name__)
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Request")

# Product-related keywords — sentiment only shows for these queries
PRODUCT_KEYWORDS = [
    "product", "headset", "headphone", "earphone", "earbuds", "buds",
    "boat", "realme", "oneplus", "bass", "wireless", "bluetooth",
    "price", "cost", "worth", "buy", "purchase", "recommend",
    "review", "rating", "quality", "sound", "battery", "compare",
    "best", "worst", "cheap", "expensive", "budget", "neckband",
    "airdopes", "rockerz", "bullets", "titanic", "wired",
]


def _is_product_related(user_input: str, answer: str) -> bool:
    """Check if the conversation is about products (not greetings/small talk).
    
    Returns True only when the user query or bot response contains
    product-related keywords, so sentiment badges don't appear
    on greetings like 'hello' or 'how are you'.
    """
    combined = (user_input + " " + answer).lower()
    return any(kw in combined for kw in PRODUCT_KEYWORDS)


def create_app():
    app = Flask(__name__)

    # Initialize components
    logger.info("Starting ShopSmart AI...")
    vector_store = DataIngestor().ingest(load_existing=True)
    rag_chain = RAGChainBuilder(vector_store).build_chain()

    # Load sentiment model (lightweight, loads in <1 second)
    try:
        sentiment_predictor = SentimentPredictor()
        logger.info("Sentiment predictor loaded")
    except FileNotFoundError as e:
        logger.warning(f"Sentiment model not found: {e}. Running without sentiment.")
        sentiment_predictor = None

    @app.route("/")
    def index():
        REQUEST_COUNT.inc()
        return render_template("index.html")

    @app.route("/get", methods=["POST"])
    def get_response():
        user_input = request.form["msg"]

        # Get RAG response
        result = rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "user-session"}}
        )
        answer = result["answer"]

        # Only enrich with sentiment when the response is product-related
        # Skip for greetings, small talk, and non-product queries
        is_product_query = (
            sentiment_predictor
            and "context" in result
            and result["context"]
            and _is_product_related(user_input, answer)
        )

        if is_product_query:
            try:
                docs = result["context"]
                reviews = [doc.page_content for doc in docs]
                predictions = sentiment_predictor.predict_batch(reviews)

                # Summarize sentiment
                sentiment_counts = {}
                for p in predictions:
                    label = p["label"]
                    sentiment_counts[label] = sentiment_counts.get(label, 0) + 1

                # Only show if there's a mix of sentiments (more interesting)
                # or if there are negative reviews worth highlighting
                summary_parts = []
                for label in ["Positive", "Neutral", "Negative"]:
                    count = sentiment_counts.get(label, 0)
                    if count > 0:
                        emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}[label]
                        summary_parts.append(f"{emoji} {label}: {count}")

                if summary_parts:
                    sentiment_line = "\n\n📊 Review Sentiment: " + " | ".join(summary_parts)
                    answer += sentiment_line
            except Exception as e:
                logger.error(f"Sentiment analysis failed: {e}")

        return answer

    @app.route("/sentiment", methods=["POST"])
    def analyze_sentiment():
        """Standalone sentiment analysis endpoint.
        
        POST /sentiment with JSON: {"text": "Great product!"}
        Returns: {"label": "Positive", "confidence": 0.85, "emoji": "😊"}
        """
        if not sentiment_predictor:
            return jsonify({"error": "Sentiment model not loaded"}), 503

        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        result = sentiment_predictor.predict(data["text"])
        return jsonify(result)

    @app.route("/metrics")
    def metrics():
        return Response(generate_latest(), mimetype="text/plain")

    return app


if __name__ == "__main__":
    app = create_app()
    print("\n✅ ShopSmart AI is ready! Open http://localhost:5000 in your browser.\n")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)