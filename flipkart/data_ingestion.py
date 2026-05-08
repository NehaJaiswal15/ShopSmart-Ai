"""
Handles data ingestion into the AstraDB vector store.

Connects to AstraDB, loads and preprocesses documents,
and uploads them with embeddings.
"""
import time
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from flipkart.data_converter import DataConverter
from flipkart.config import Config
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

DATA_FILE = "data/flipkart_product_review.csv"
COLLECTION_NAME = "flipkart_database"


class DataIngestor:
    def __init__(self):
        logger.info(f"Initializing embeddings: {Config.EMBEDDING_MODEL}")
        self.embedding = HuggingFaceEndpointEmbeddings(model=Config.EMBEDDING_MODEL)

        logger.info(f"Connecting to AstraDB collection: {COLLECTION_NAME}")
        self.vstore = AstraDBVectorStore(
            embedding=self.embedding,
            collection_name=COLLECTION_NAME,
            api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
            token=Config.ASTRA_DB_APPLICATION_TOKEN,
            namespace=Config.ASTRA_DB_KEYSPACE
        )
        logger.info("AstraDB connection established")

    def ingest(self, load_existing=True):
        """Ingest data into the vector store.

        Args:
            load_existing: If True, skip ingestion and return existing store.
                           If False, convert + upload fresh documents.

        Returns:
            The AstraDB vector store instance.
        """
        if load_existing:
            logger.info("Using existing vector store (load_existing=True)")
            return self.vstore

        try:
            logger.info("Starting fresh data ingestion...")
            start_time = time.time()

            docs = DataConverter(DATA_FILE).convert()
            logger.info(f"Prepared {len(docs)} documents for upload")

            self.vstore.add_documents(docs)

            elapsed = time.time() - start_time
            logger.info(
                f"Ingestion complete: {len(docs)} documents uploaded "
                f"in {elapsed:.1f} seconds"
            )
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise CustomException("Data ingestion failed", e)

        return self.vstore
