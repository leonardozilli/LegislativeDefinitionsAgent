import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from LegalDefAgent.src.settings import settings
from LegalDefAgent.src.utils import setup_logging

setup_logging()


def get_xml_mapping(xml_file: Path, dataset: str):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        namespace = settings.DB_CONFIG.NAMESPACES[dataset]
        
        definitions_el = root.find('.//akn:FRBRWork', namespace)
        if not definitions_el:
            return None
            
    except Exception as e:
        logging.error(f"Error processing file {xml_file}: {e}")
        return None


def main():
    config = settings.DB_CONFIG
    data_dir = Path(config.XML_DATA_DIR)

    extracted, errors, total = 0, 0, 0

    for dataset in config.DATASETS:
        mapping_list = []
        logging.info(f"Mapping XML files in dataset {dataset}...")
        target = config.XML_DATA_DIR / dataset
        for file in target.rglob('*.xml'):
            total += 1
            try:
                mapping = get_xml_mapping(file, dataset)
                if mapping:
                    extracted += 1
                    mapping_list.extend(mapping)
            except Exception as e:
                logging.error(f"Error processing {file}: {e}")
    
        break
    logging.info(f'Extracted definitions from {extracted} files. {errors} errors out of {total} files.')j<<



if __name__ == "__main__":
    main()