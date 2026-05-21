"""
Unit tests for the Flask application and telemetry instrumentation.

Covers:
- Base index page rendering
- REST health check endpoint (/health)
- Prometheus metrics endpoint (/metrics)
- Chatbot get response with and without product context / sentiment
- Sentiment standalone POST endpoint

Run:
    python -m pytest tests/test_app.py -v
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_components():
    """Mock out heavy ML models and external database/API components."""
    with (
        patch("app.DataIngestor") as mock_ingestor,
        patch("app.RAGChainBuilder") as mock_builder,
        patch("app.SentimentPredictor") as mock_predictor,
    ):
        # Mock vector store
        mock_vs = MagicMock()
        mock_ingestor.return_value.ingest.return_value = mock_vs

        # Mock RAG chain
        mock_chain = MagicMock()
        mock_builder.return_value.build_chain.return_value = mock_chain

        # Mock RAG chain invocation output
        mock_chain.invoke.return_value = {
            "answer": "This is a great boat headset with superb bass quality.",
            "context": [MagicMock(page_content="Review: amazing sound quality and boat base")],
        }

        # Mock SentimentPredictor instance
        mock_pred_instance = MagicMock()
        mock_predictor.return_value = mock_pred_instance

        # Standalone predictions list for review sentiment mapping
        mock_pred_instance.predict.return_value = {"label": "Positive", "score": 0.98}
        mock_pred_instance.predict_batch.return_value = [{"label": "Positive", "confidence": 0.98}]

        yield {
            "ingestor": mock_ingestor,
            "builder": mock_builder,
            "chain": mock_chain,
            "predictor_class": mock_predictor,
            "predictor_instance": mock_pred_instance,
        }


@pytest.fixture
def client(mock_components):
    """Test client fixture for testing Flask routes."""
    # Import app inside fixture to ensure mock patches are active during Flask setup
    from app import create_app

    flask_app = create_app()
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as test_client:
        yield test_client


def test_index_route(client):
    """Test that the index route renders index.html."""
    with patch("app.render_template", return_value="Index Mock Page") as mock_render:
        res = client.get("/")
        assert res.status_code == 200
        assert res.data.decode("utf-8") == "Index Mock Page"
        mock_render.assert_called_once_with("index.html")


def test_health_endpoint_healthy(client):
    """Test the /health endpoint when all components are active."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert data["components"]["astradb"] == "healthy"
    assert data["components"]["groq"] == "healthy"
    assert data["components"]["sentiment_model"] == "healthy"


def test_metrics_endpoint(client):
    """Test that the /metrics endpoint outputs Prometheus metric formats."""
    res = client.get("/metrics")
    assert res.status_code == 200
    content = res.data.decode("utf-8")
    assert "http_requests_total" in content
    assert "response_latency_seconds" in content
    assert "retrieved_docs_count" in content


def test_get_response_product_related(client, mock_components):
    """Test RAG invocation and sentiment enrichment when asking about product attributes."""
    res = client.post("/get", data={"msg": "Is the boat headset worth the price?"})
    assert res.status_code == 200
    answer = res.data.decode("utf-8")

    # Assert base answer is returned
    assert "This is a great boat headset" in answer

    # Assert sentiment enrichment is present in the response
    assert "Review Sentiment" in answer
    assert "Positive: 1" in answer

    # Assert RAG was invoked with correct arguments
    mock_components["chain"].invoke.assert_called_once_with(
        {"input": "Is the boat headset worth the price?"}, config={"configurable": {"session_id": "user-session"}}
    )


def test_sentiment_endpoint(client, mock_components):
    """Test that standalone sentiment prediction works through the REST API."""
    res = client.post("/sentiment", json={"text": "Very bad quality and cost."})
    assert res.status_code == 200
    data = res.get_json()

    assert data["label"] == "Positive"  # Match our mock value
    mock_components["predictor_instance"].predict.assert_called_once_with("Very bad quality and cost.")
