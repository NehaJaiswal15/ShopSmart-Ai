"""
MLflow Experiment Tracking Helper

Provides central utilities to log training and evaluation runs to a local
MLflow instance (using the file system database in `./mlruns`).
"""

import os
import mlflow
from utils.logger import get_logger

logger = get_logger(__name__)

# Use a local filesystem-based tracking URI (no database setup required)
TRACKING_URI = "file:./mlruns"
mlflow.set_tracking_uri(TRACKING_URI)


def get_or_create_experiment(experiment_name: str) -> str:
    """Get an existing experiment or create a new one by name.

    Args:
        experiment_name: Name of the experiment.

    Returns:
        The experiment ID string.
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            return experiment.experiment_id

        return mlflow.create_experiment(experiment_name)
    except Exception as e:
        logger.warning(f"Error checking/creating MLflow experiment '{experiment_name}': {e}. Using default.")
        return "0"


def log_rag_eval_run(
    retrieval_strategy: str,
    k: int,
    prompt_template_name: str,
    metrics: dict,
    run_name: str = None,
    report_path: str = None,
):
    """Log a RAG evaluation run to MLflow.

    Args:
        retrieval_strategy: E.g., 'hybrid', 'vector', 'bm25'
        k: Number of retrieved documents
        prompt_template_name: Name/identifier of the system prompt
        metrics: Dictionary of evaluation metrics (e.g., RAGAS scores)
        run_name: Optional name for this specific run
        report_path: Optional path to the full report JSON to log as an artifact
    """
    try:
        experiment_id = get_or_create_experiment("RAG_Evaluation")

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            # Log hyperparameters
            mlflow.log_param("retrieval_strategy", retrieval_strategy)
            mlflow.log_param("k", k)
            mlflow.log_param("prompt_template_name", prompt_template_name)

            # Log metrics
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(metric_name, value)

            # Log full report JSON if provided
            if report_path and os.path.exists(report_path):
                mlflow.log_artifact(report_path)

            logger.info(
                f"Successfully logged RAG eval run '{run_name or 'unnamed'}' to MLflow (experiment: RAG_Evaluation)"
            )
    except Exception as e:
        logger.error(f"Failed to log RAG eval run to MLflow: {e}")
