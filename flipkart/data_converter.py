"""
Converts raw CSV product review data into LangChain Document objects
for ingestion into the vector store.

Applies text preprocessing and validation to ensure data quality.
"""

import os
import pandas as pd
from langchain_core.documents import Document
from flipkart.data_preprocessing import clean_text, is_valid_review, validate_dataframe_schema
from utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["product_title", "rating", "summary", "review"]


class DataConverter:
    def __init__(self, file_path: str, price_file: str = None):
        self.file_path = file_path
        self.price_file = price_file or os.path.join(os.path.dirname(file_path), "product_prices.csv")

    def _load_prices(self):
        """Load product prices from the price CSV into a lookup dict."""
        if os.path.exists(self.price_file):
            price_df = pd.read_csv(self.price_file)
            price_map = dict(zip(price_df["product_title"], price_df["price"]))
            logger.info(f"Loaded prices for {len(price_map)} products")
            return price_map
        logger.warning(f"Price file not found: {self.price_file}")
        return {}

    def convert(self):
        """Convert CSV data to cleaned, validated LangChain Documents.

        Pipeline:
            1. Load CSV and validate schema
            2. Drop rows with NaN reviews
            3. Clean review text (HTML, whitespace, URLs)
            4. Filter out invalid reviews (too short, gibberish)
            5. Merge price data
            6. Build Document objects with rich metadata

        Returns:
            List of LangChain Document objects.
        """
        logger.info(f"Loading data from: {self.file_path}")
        df = pd.read_csv(self.file_path)

        # Step 1: Schema validation
        validate_dataframe_schema(df, REQUIRED_COLUMNS)

        df = df[REQUIRED_COLUMNS]
        total_rows = len(df)

        # Step 2: Handle NaN values
        nan_count = df["review"].isna().sum()
        if nan_count > 0:
            logger.warning(f"Dropping {nan_count} rows with NaN reviews")
        df = df.dropna(subset=["review"])

        # Fill NaN summaries with empty string
        df["summary"] = df["summary"].fillna("")

        # Step 3: Clean text
        df["review"] = df["review"].apply(clean_text)
        df["summary"] = df["summary"].apply(clean_text)

        # Step 4: Filter invalid reviews
        valid_mask = df["review"].apply(is_valid_review)
        skipped = (~valid_mask).sum()
        if skipped > 0:
            logger.warning(f"Filtered out {skipped} reviews that were too short or invalid")
        df = df[valid_mask]

        # Step 5: Load prices
        price_map = self._load_prices()

        # Step 6: Build documents
        docs = []
        for _, row in df.iterrows():
            product = row["product_title"]
            price = price_map.get(product, "N/A")

            content = (
                f"Product: {product}\n"
                f"Price: ₹{price}\n"
                f"Rating: {row['rating']}/5\n"
                f"Summary: {row['summary']}\n"
                f"Review: {row['review']}"
            )

            docs.append(
                Document(
                    page_content=content,
                    metadata={"product_name": product, "price": price, "rating": int(row["rating"])},
                )
            )

        logger.info(
            f"Conversion complete: {total_rows} rows -> {len(docs)} documents (dropped {total_rows - len(docs)} rows)"
        )
        return docs
