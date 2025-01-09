import argparse
import logging
import pickle
from typing import Optional, Any, List
import os

from LegalDefAgent.src.utils import setup_logging
from LegalDefAgent.src.settings import settings
from LegalDefAgent.src.retriever.extractor import DefinitionExtractor
from LegalDefAgent.src.retriever.embedder import VectorDBBuilder

setup_logging()

def save_definitions_list_to_pickle(df, output_file='definitions.pkl'):
    definitions_list = df['definition_text'].to_list()
    with open(output_file, 'wb') as f:
        pickle.dump(definitions_list, f)
    print(f"Definitions list saved to {output_file}")


def main(pickle=False):
    extractor = DefinitionExtractor()
    processed_df = extractor.extract_and_filter()
    if pickle:
        save_definitions_list_to_pickle(processed_df)
    else:
        output_file = os.path.join(settings.DB_CONFIG.DEFINITIONS_OUTPUT_DIR, "definitions.parquet")
        if not os.path.exists(settings.DB_CONFIG.DEFINITIONS_OUTPUT_DIR):
            os.makedirs(settings.DB_CONFIG.DEFINITIONS_OUTPUT_DIR)
        processed_df.write_parquet(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the vector database of legal definitions.")
    parser.add_argument('--pickle', action='store_true', help='Only create the list of definitions and save to a pickle file to perform the embedding phase later (or on another machine, e.g. Google Colab).')
    #parser.add_argument('--datasets', nargs='+', help='Optional list of datasets to process. If None, uses all configured datasets.')

    args = parser.parse_args()

    main(pickle=args.pickle)