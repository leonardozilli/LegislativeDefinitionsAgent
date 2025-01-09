import argparse
import logging
import pickle
from typing import Optional, Any, List
import polars as pl

from LegalDefAgent.src.utils import setup_logging
from LegalDefAgent.src.settings import settings
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

def save_definitions_list_to_pickle(output_file='definitions.pkl'):
    extractor = DefinitionExtractor()
    processed_df = extractor.extract_and_filter()
    definitions_list = processed_df['definition_text'].to_list()
    with open(output_file, 'wb') as f:
        pickle.dump(definitions_list, f)
    print(f"Definitions list saved to {output_file}")

def populate_vec_db(definitions_df, embeddings):
    vec_builder = VectorDBBuilder()
    df = pl.read_parquet(definitions_df)
    embeddings = pickle.load(open(embeddings, 'rb'))
    vec_builder.build_vector_db(df=df, defs_embeddings=embeddings)
    
def main(definitions_list, embeddings):
    populate_vec_db(definitions_list, embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the vector database of legal definitions.")
    parser.add_argument('--def_list', type=str, required=True, help='Path to the list of definitions to insert into the database.')
    parser.add_argument('--embeddings', type=str, required=True, help='Path to the embeddings of the definitions.')
    #parser.add_argument('--datasets', nargs='+', help='Optional list of datasets to process. If None, uses all configured datasets.')
    args = parser.parse_args()

    main(definitions_list=args.def_list, embeddings=args.embeddings)