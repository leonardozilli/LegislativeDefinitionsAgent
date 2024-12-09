import argparse
import logging
import pickle
from typing import Optional, Any, List

from LegalDefAgent.src.utils import setup_logging
from LegalDefAgent.src.retriever.extractor import DefinitionExtractor
from LegalDefAgent.src.retriever.embedder import VectorDBBuilder

setup_logging()


def extract_process_embed_defs(
    #embedding_function: Any,
    datasets: Optional[List[str]] = None,
) -> None:
    """
    Build the vector database of legal definitions.

    Args:
        embedding_function: Function to generate embeddings
        datasets: Optional list of datasets to process. If None, uses all configured datasets.
    """
    try:
        # Extract and process definitions
        extractor = DefinitionExtractor()
        processed_df = extractor.extract_and_filter()

        # Build vector database
        db_builder = VectorDBBuilder()
        db_builder.build_vector_db(processed_df)

    except Exception as e:
        logging.error(f"Error building database: {e}")
        raise

def save_definitions_list_to_pickle(datasets=None, output_file='definitions.pkl'):
    extractor = DefinitionExtractor()
    processed_df = extractor.extract_and_filter(datasets=datasets)
    definitions_list = processed_df['definition_text'].tolist()
    with open(output_file, 'wb') as f:
        pickle.dump(definitions_list, f)
    print(f"Definitions list saved to {output_file}")

def insert_definitions_into_db(defs_df, defs_embeddings):
    pass

def main(only_definitions=False, insert_defs=False, datasets=None):
    if only_definitions:
        save_definitions_list_to_pickle(datasets=datasets)
    elif insert_defs:
        pass
    else:
        extract_process_embed_defs(datasets=datasets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the vector database of legal definitions.")
    parser.add_argument('--definitions_list', action='store_true', help='Only create the list of definitions and save to a pickle file.')
    parser.add_argument('--datasets', nargs='+', help='Optional list of datasets to process. If None, uses all configured datasets.')

    args = parser.parse_args()

    main(only_definitions=args.only_definitions, datasets=args.datasets)