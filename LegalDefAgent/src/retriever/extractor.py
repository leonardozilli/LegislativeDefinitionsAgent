import xml.etree.ElementTree as ET
import re
import csv
from pathlib import Path
import logging
import polars as pl
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from ..config import DB_CONFIG


logger = logging.getLogger(__name__)

class DefinitionExtractor:
    def __init__(self):
        self.config = DB_CONFIG
        self.data_dir = Path(self.config['XML_DATA_DIR'])
        #self.output_dir = Path(self.config['OUTPUT_DIR'])
        self.definitions_output_dir = Path(self.config['DEFINITIONS_OUTPUT_DIR'])
        self.namespaces = self.config['NAMESPACES']

    def extract_and_filter(self) -> pl.DataFrame:
        """Extract definitions from all datasets directly into a DataFrame."""
        all_definitions = []
        extracted, errors, total = 0, 0, 0
        
        for dataset in self.config['DATASETS']:
            logger.info(f"Extracting definitions from XML files in dataset {dataset}...")
            target = self.data_dir / dataset
            for file in target.rglob('*.xml'):
                total += 1
                try:
                    definitions = self.parse_xml(file, dataset)
                    if definitions:
                        extracted += 1
                        all_definitions.extend(definitions)
                except Exception as e:
                    errors += 1
                    logger.error(f"Error processing {file}: {e}")
        
        logger.info(f'Extracted definitions from {extracted} files. {errors} errors out of {total} files.')
        
        # Create DataFrame and process immediately
        if not all_definitions:
            raise ValueError("No definitions were extracted")
            
        df = pl.DataFrame(all_definitions)
        
        # Process and transform the data
        processed_df = (
            df.with_columns(
                pl.when(
                    pl.col("definendum").is_null() | pl.col("definiens").is_null()
                ).then(
                    pl.col("full_definition")
                ).otherwise(
                    pl.concat_str(
                        [pl.col("definendum"), pl.col("definiens")],
                        separator=": ",
                    )
                ).alias("joined_definition")
            )
            .filter(pl.col('full_definition').str.len_chars() < self.config['MAX_DEFINITION_LENGTH'])
            .select(
                pl.col('joined_definition').alias('definition_text'),
                pl.col('def_n'),
                pl.col('provenance').alias('dataset'),
                pl.col('document').alias('document_id'),
                pl.col('references'),
            )
            #.with_columns([
                #pl.col('references').map_elements(eval, return_dtype=pl.List(pl.String))
            #])
            .with_row_index('id')
        )
        
        return processed_df
        
    def parse_xml(self, xml_file: Path, dataset: str) -> Optional[List[Dict]]:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            namespace = self.namespaces[dataset]
            
            definitions_el = root.findall('.//akn:definitions', namespace)
            if not definitions_el:
                return None
                
            definitions = []
            for definition in root.findall('.//akn:definition', namespace):
                try:
                    definendum, definiens, references, full_def = self._parse_definition(
                        definition, root, namespace
                    )
                    definitions.append({
                        'def_n': definition.find('.//akn:definitionHead', namespace).attrib.get('href', ''),
                        'label': definition.find('.//akn:definitionHead', namespace).attrib.get('refersTo', ''),
                        'definendum': self._clean_definendum(definendum),
                        'definiens': self._clean_definiens(definiens),
                        'full_definition': self._clean_full_def(full_def),
                        'references': references,
                        'provenance': dataset,
                        'document': xml_file.name
                    })
                except Exception as e:
                    #logger.error(f"Error parsing definition in {xml_file}: {e}")
                    pass
                    
            return definitions
        except Exception as e:
            logger.error(f"Error processing file {xml_file}: {e}")
            return None

    def _parse_definition(self, definition, root, namespace) -> Tuple[str, str, List[str], str]:
        """Extract components from a single definition."""
        definition_head = definition.find('.//akn:definitionHead', namespace)
        definition_body_elements = definition.findall('.//akn:definitionBody', namespace)
        
        definendum_id = definition_head.attrib.get('href', '').lstrip('#')
        definendum = root.find(f".//akn:def[@eId='{definendum_id}']", namespace).text
        
        try:
            full_def = "".join(root.find(f".//*[@defines='#{definendum_id}']").itertext())
        except Exception:
            full_def = None
            
        definiens = []
        references = []
        for body in definition_body_elements:
            body_text, body_refs = self._extract_body_and_references(body, root, namespace)
            definiens.append(body_text)
            references.extend(body_refs)
            
        return definendum, ' '.join(definiens), references, full_def

    @staticmethod
    def _extract_body_and_references(body, root, namespace) -> Tuple[str, List[str]]:
        """Extract text and references from definition body."""
        body_id = body.attrib.get('href', '').lstrip('#')
        body_element = root.find(f".//akn:defBody[@eId='{body_id}']", namespace)
        body_text = ''.join(body_element.itertext())
        references = [
            ref.attrib.get('href', '') 
            for ref in body_element.findall('.//akn:ref', namespace)
        ]
        return body_text, references

    @staticmethod
    def _clean_definendum(text: str) -> str:
        text = text.strip()
        if text.startswith(("«", "'", "\"")) and text.endswith(("»", "'", "\"")):
            text = text[1:-1]
        return text.strip()

    @staticmethod
    def _clean_definiens(text: str) -> str:
        if text.startswith((':',',')):
            text = text.lstrip(':,')
        return re.sub(r'\s+', ' ', text.strip().replace('\n', ' '))

    @staticmethod
    def _clean_full_def(text: str) -> str:
        if text is None:
            return ""
        return re.sub(r'\s+', ' ', text.strip().replace('\n', ' ').replace('""', ''))

    def extract_all(self) -> None:
        """Extract definitions from all datasets."""
        extracted, errors, total = 0, 0, 0
        self.definitions_output_dir.mkdir(parents=True, exist_ok=True)
        
        for dataset in self.config['DATASETS']:
            target = Path(self.data_dir / dataset)
            for file in target.rglob('*.xml'):
                total += 1
                try:
                    definitions = self.parse_xml(file, dataset)
                    if definitions:
                        extracted += 1
                        output_file = self.definitions_output_dir / file.name.replace('.xml', '.tsv')
                        self._save_definitions(definitions, output_file)
                except Exception as e:
                    errors += 1
                    logger.error(f"Error processing {file}: {e}")
                    
        logger.info(f'Extracted definitions from {extracted} files. {errors} errors out of {total} files.')

    def _save_definitions(self, definitions: List[Dict], output_file: Path) -> None:
        """Save extracted definitions to TSV file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(
                csvfile, 
                fieldnames=definitions[0].keys(), 
                delimiter="\t", 
                quoting=csv.QUOTE_MINIMAL
            )
            writer.writeheader()
            for row in definitions:
                writer.writerow(row)
