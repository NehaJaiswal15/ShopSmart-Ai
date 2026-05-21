"""
Unit tests for the data preprocessing pipeline.

Tests cover:
- Text cleaning (HTML, URLs, whitespace)
- Review validation (length, meaningful content)
- Schema validation (required columns)
- Edge cases (empty strings, None values, special characters)

Run:
    python -m pytest tests/test_data_preprocessing.py -v
"""

import pytest
import pandas as pd
from flipkart.data_preprocessing import clean_text, is_valid_review, validate_dataframe_schema


REQUIRED_COLUMNS = ["product_title", "rating", "summary", "review"]


# ── clean_text Tests ───────────────────────────────────────────


class TestCleanText:
    """Tests for the clean_text function."""

    def test_basic_strip(self):
        """Leading/trailing whitespace should be stripped."""
        assert clean_text("  Great Product!  ") == "Great Product!"

    def test_html_tag_removal(self):
        """HTML tags should be stripped."""
        result = clean_text("<b>Bold text</b>")
        assert "<b>" not in result
        assert "Bold text" in result

    def test_html_entity_decode(self):
        """HTML entities like &amp; should be decoded."""
        assert "&" in clean_text("Tom &amp; Jerry")

    def test_url_removal(self):
        """URLs should be removed from text."""
        result = clean_text("Check https://example.com for details")
        assert "https" not in result
        assert "example.com" not in result

    def test_extra_whitespace(self):
        """Multiple spaces should be collapsed to single space."""
        result = clean_text("Too   many    spaces")
        assert "  " not in result
        assert result == "Too many spaces"

    def test_newlines_to_spaces(self):
        """Newlines should be replaced with spaces."""
        result = clean_text("Line one\nLine two\rLine three")
        assert "\n" not in result
        assert "\r" not in result

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert clean_text("") == ""

    def test_non_string_input(self):
        """Non-string input should return empty string."""
        assert clean_text(None) == ""
        assert clean_text(12345) == ""
        assert clean_text(3.14) == ""

    def test_special_characters_preserved(self):
        """Punctuation should be kept."""
        result = clean_text("Good product! Worth the price.")
        assert "!" in result
        assert "." in result


# ── is_valid_review Tests ──────────────────────────────────────


class TestIsValidReview:
    """Tests for the is_valid_review function."""

    def test_valid_review(self):
        """Normal review should pass validation."""
        assert is_valid_review("This is a perfectly good review text") is True

    def test_too_short(self):
        """Reviews under 10 chars should fail."""
        assert is_valid_review("Good") is False
        assert is_valid_review("OK prod") is False

    def test_empty_string(self):
        """Empty string should fail."""
        assert is_valid_review("") is False

    def test_at_min_length(self):
        """Review at exactly 10 chars should pass (>= not >)."""
        assert is_valid_review("Great prod") is True  # 10 chars with letters

    def test_only_numbers(self):
        """Review with only numbers should fail (no meaningful text)."""
        assert is_valid_review("1234567890123") is False

    def test_only_punctuation(self):
        """Review with only punctuation should fail."""
        assert is_valid_review("!!!!!!!!!!!") is False

    def test_mixed_content_valid(self):
        """Review with letters + numbers should pass."""
        assert is_valid_review("5 stars product amazing") is True


# ── validate_dataframe_schema Tests ────────────────────────────


class TestValidateSchema:
    """Tests for the validate_dataframe_schema function."""

    def test_valid_schema(self):
        """DataFrame with all required columns should not raise."""
        df = pd.DataFrame(
            {
                "product_title": ["Product A"],
                "rating": [4],
                "summary": ["Good"],
                "review": ["This is a good product"],
            }
        )
        # Should not raise
        validate_dataframe_schema(df, REQUIRED_COLUMNS)

    def test_missing_column_raises(self):
        """DataFrame missing a required column should raise ValueError."""
        df = pd.DataFrame(
            {
                "product_title": ["Product A"],
                "rating": [4],
            }
        )
        with pytest.raises(ValueError, match="missing required columns"):
            validate_dataframe_schema(df, REQUIRED_COLUMNS)

    def test_empty_dataframe_correct_columns(self):
        """Empty DataFrame with correct columns should not raise."""
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        validate_dataframe_schema(df, REQUIRED_COLUMNS)

    def test_extra_columns_ok(self):
        """Extra columns beyond required should still pass."""
        df = pd.DataFrame(
            {
                "product_title": ["Product A"],
                "rating": [4],
                "summary": ["Good"],
                "review": ["This is a good product"],
                "extra_col": ["extra data"],
            }
        )
        validate_dataframe_schema(df, REQUIRED_COLUMNS)


# ── Integration Tests ──────────────────────────────────────────


class TestIntegration:
    """End-to-end tests combining clean_text and is_valid_review."""

    def test_clean_then_validate(self):
        """Cleaned HTML review should pass validation."""
        raw = "  <b>This is a great product</b> with good quality!  "
        cleaned = clean_text(raw)
        assert is_valid_review(cleaned) is True

    def test_html_only_review_fails(self):
        """Review that's only HTML tags should fail after cleaning."""
        raw = "<br><br><hr>"
        cleaned = clean_text(raw)
        assert is_valid_review(cleaned) is False

    def test_url_only_review_fails(self):
        """Review that's only a URL should fail after cleaning."""
        raw = "https://www.example.com/very/long/url/path"
        cleaned = clean_text(raw)
        assert is_valid_review(cleaned) is False
