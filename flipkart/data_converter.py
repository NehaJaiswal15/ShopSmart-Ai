import os
import pandas as pd
from langchain_core.documents import Document

class DataConverter:
    def __init__(self, file_path: str, price_file: str = None):
        self.file_path = file_path
        self.price_file = price_file or os.path.join(
            os.path.dirname(file_path), "product_prices.csv"
        )

    def _load_prices(self):
        """Load product prices from the price CSV into a lookup dict."""
        if os.path.exists(self.price_file):
            price_df = pd.read_csv(self.price_file)
            return dict(zip(price_df["product_title"], price_df["price"]))
        return {}

    def convert(self):
        df = pd.read_csv(self.file_path)[["product_title", "rating", "summary", "review"]]
        price_map = self._load_prices()

        docs = []
        for _, row in df.iterrows():
            product = row["product_title"]
            price = price_map.get(product, "N/A")

            # Include price and rating in page_content so the LLM sees it
            content = (
                f"Product: {product}\n"
                f"Price: ₹{price}\n"
                f"Rating: {row['rating']}/5\n"
                f"Summary: {row['summary']}\n"
                f"Review: {row['review']}"
            )

            docs.append(Document(
                page_content=content,
                metadata={
                    "product_name": product,
                    "price": price,
                    "rating": int(row["rating"])
                }
            ))

        return docs