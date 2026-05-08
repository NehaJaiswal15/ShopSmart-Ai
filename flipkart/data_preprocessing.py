"""
Text preprocessing pipeline for product review data.

Handles cleaning, normalization, and validation of review text
before it enters the vector store or ML models.
"""
import re
import html
from utils.logger import get_logger

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    """Clean and normalize review text.
    
    Steps:
        1. Decode HTML entities (&amp; -> &, etc.)
        2. Strip HTML tags (<br>, <b>, etc.)
        3. Normalize whitespace (multiple spaces/newlines -> single space)
        4. Strip leading/trailing whitespace
    
    Args:
        text: Raw review text.
    
    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""
    
    # Decode HTML entities (e.g., &amp; -> &, &lt; -> <)
    text = html.unescape(text)
    
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    
    # Normalize whitespace (tabs, newlines, multiple spaces -> single space)
    text = re.sub(r"\s+", " ", text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def is_valid_review(text: str, min_length: int = 10) -> bool:
    """Check if a review is meaningful enough to include.
    
    Args:
        text: Cleaned review text.
        min_length: Minimum character length to consider valid.
    
    Returns:
        True if review meets quality threshold.
    """
    if not text:
        return False
    if len(text) < min_length:
        return False
    # Reject reviews that are just punctuation or numbers
    if not re.search(r"[a-zA-Z]{3,}", text):
        return False
    return True


def validate_dataframe_schema(df, required_columns: list) -> None:
    """Validate that a DataFrame has the expected columns.
    
    Raises ValueError with a clear message if columns are missing.
    This catches data issues at ingestion time instead of producing
    silent errors downstream.
    
    Args:
        df: pandas DataFrame to validate.
        required_columns: List of column names that must be present.
    
    Raises:
        ValueError: If any required column is missing.
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    logger.info(f"Schema validation passed. Columns: {list(df.columns)}")
