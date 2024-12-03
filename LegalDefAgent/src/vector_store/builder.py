import logging
from pathlib import Path
from typing import Optional, Any, List
from ..config import DB_CONFIG
from .extractor import DefinitionExtractor
from .processor import DefinitionProcessor
from .embedder import VectorDBBuilder

logger = logging.getLogger(__name__)


def build_database(
    embedding_function: Any,
    datasets: Optional[List[str]] = None,
) -> None:
    """
    Build the vector database of legal definitions.

    Args:
        embedding_function: Function to generate embeddings
        datasets: Optional list of datasets to process. If None, uses all configured datasets.
    """
    try:
        # Extract definitions
        logger.info("Extracting definitions from XML files...")
        extractor = DefinitionExtractor()
        extractor.extract_all()

        # Process definitions
        logger.info("Processing extracted definitions...")
        processor = DefinitionProcessor()
        tsv_files = list(Path(DB_CONFIG['OUTPUT_DIR']).glob('*.tsv'))
        processed_df = processor.process_definitions(tsv_files)

        # Build vector database
        logger.info("Building vector database...")
        db_builder = VectorDBBuilder(embedding_function)
        db_builder.build_vector_db(processed_df)

        logger.info("Vector database built successfully!")

    except Exception as e:
        logger.error(f"Error building database: {e}")
        raise