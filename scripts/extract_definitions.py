import argparse
import logging
import pickle
import os

from LegalDefAgent.src.utils import setup_logging
from LegalDefAgent.src.settings import settings
from LegalDefAgent.src.retriever.extractor import DefinitionExtractor

setup_logging()


def save_definitions_list_to_pickle(df, output_file='definitions_list.pkl'):
    logging.info("Saving definitions list to pickle file...")
    definitions_list = df['definition_text'].to_list()
    with open(output_file, 'wb') as f:
        pickle.dump(definitions_list, f)
    logging.info(f"Definitions list saved to {output_file}")


def main(pickle=False):
    extractor = DefinitionExtractor()
    processed_df = extractor.extract_and_filter()
    if pickle:
        save_definitions_list_to_pickle(processed_df)
    else:
        output_file = os.path.join(
            settings.DB_CONFIG.DEFINITIONS_OUTPUT_DIR, "definitions.csv")
        if not os.path.exists(settings.DB_CONFIG.DEFINITIONS_OUTPUT_DIR):
            os.makedirs(settings.DB_CONFIG.DEFINITIONS_OUTPUT_DIR)
        processed_df.write_csv(output_file)
        logging.info(f"{processed_df.__len__()} Definitions saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the vector database of legal definitions.")
    parser.add_argument('--pickle', action='store_true',
                        help='Only create the list of definitions and save to a pickle file to perform the embedding phase later (or on another machine, e.g. Google Colab).')

    args = parser.parse_args()

    main(pickle=args.pickle)
