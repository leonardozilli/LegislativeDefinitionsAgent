import argparse
import pickle
import xml.etree.ElementTree as ET
import re
from pathlib import Path
import logging
import polars as pl
from typing import List, Dict, Optional, Tuple

from LegalDefAgent.src.utils import setup_logging
from LegalDefAgent.src.settings import settings
from LegalDefAgent.src.existdb import existdb_handler

setup_logging()


class DefinitionExtractor:
    def __init__(self):
        self.config = settings.DB_CONFIG
        self.data_dir = Path(self.config.XML_DATA_DIR)
        self.definitions_output_dir = Path(self.config.DEFINITIONS_OUTPUT_DIR)
        self.namespaces = self.config.NAMESPACES
        self.existdb_handler = existdb_handler

    def extract_and_filter(self) -> pl.DataFrame:
        """Extract definitions from all datasets directly into a DataFrame."""
        all_definitions = []
        extracted, errors, total = 0, 0, 0

        for dataset in self.config.DATASETS:
            logging.info(
                f"Extracting definitions from XML files in dataset {dataset}...")
            target = self.data_dir / dataset
            for file in target.rglob('*.xml'):
                total += 1
                try:
                    data = self.parse_xml(file, dataset)
                    if data:
                        extracted += 1
                        all_definitions.extend(data)
                except Exception as e:
                    errors += 1
                    logging.error(f"Error processing {file}: {e}")

        logging.info(f'Extracted definitions from {extracted} files. {errors} errors out of {total} files.')

        if not all_definitions:
            raise ValueError("No definitions were extracted")

        df = pl.DataFrame(all_definitions)

        processed_df = (
            df
            .with_columns(
                pl.when(pl.col("definendum").is_null() | pl.col("definiens").is_null())
                .then(pl.col("full_definition"))
                .otherwise(pl.concat_str([pl.col("definendum"), pl.col("definiens")], separator=": "))
                .alias("joined_definition")
            )
            .with_columns(
                pl.struct(pl.col("joined_definition"), pl.col("references"), pl.col("provenance"))
                .map_elements(self.append_refs, return_dtype=pl.String)
                .alias("def_with_refs")
            )
            .filter(pl.col("def_with_refs").str.len_chars() < 5000)
            .with_columns(pl.col("def_with_refs").str.split(" ").list.len().alias("word_count"))
            .filter((pl.col("word_count") > 3) & (pl.col("word_count") < 500))
            .select(
                pl.col("def_with_refs").alias("definition_text"),
                pl.col("def_n"),
                pl.col("label"),
                pl.col("provenance").alias("dataset"),
                pl.col("document").alias("document_id"),
                pl.col("frbr_work"),
                pl.col("frbr_expression"),
                pl.col("keywords"),
            )
            .with_row_index("id")
        )

        return processed_df

    def parse_xml(self, xml_file: Path, dataset: str) -> Optional[List[Dict]]:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            namespace = self.namespaces[dataset]

            frbr_work = root.find('.//akn:FRBRWork', namespace).find(
                './/akn:FRBRthis', namespace).attrib.get('value', '')
            frbr_expression = root.find('.//akn:FRBRExpression', namespace).find(
                './/akn:FRBRthis', namespace).attrib.get('value', '')
            keywords = root.findall('.//akn:keyword', namespace)

            definitions_el = root.findall('.//akn:definitions', namespace)
            if not definitions_el:
                return None

            definitions = []
            for definition in root.findall('.//akn:definition', namespace):
                try:
                    definendum, definiens, references, full_def = self._parse_definition(
                        definition, root, namespace
                    )
                    if definendum:
                        definitions.append({
                            'def_n': definition.find('.//akn:definitionHead', namespace).attrib.get('href', ''),
                            'label': definition.find('.//akn:definitionHead', namespace).attrib.get('refersTo', ''),
                            'definendum': self._clean_definendum(definendum),
                            'definiens': self._clean_definiens(definiens),
                            'full_definition': self._clean_full_def(full_def),
                            'references': references,
                            'provenance': dataset,
                            'document': xml_file.name,
                            'frbr_work': frbr_work,
                            'frbr_expression': frbr_expression,
                            'keywords': ', '.join([kw.attrib.get('showAs', '') for kw in keywords]),
                        })
                except Exception as e:
                    logging.error(f"Error parsing definition in {
                                  xml_file}: {e}")
                    pass

            return definitions
        except Exception as e:
            logging.error(f"Error processing file {xml_file}: {e}")
            return None

    def _parse_definition(self, definition, root, namespace) -> Tuple[str, str, List[str], str]:
        """Extract components from a single definition."""
        definition_head = definition.find('.//akn:definitionHead', namespace)
        definition_body_elements = definition.findall(
            './/akn:definitionBody', namespace)

        definendum_id = definition_head.attrib.get('href', '').lstrip('#')
        try:
            definendum = root.find(
                f".//akn:def[@eId='{definendum_id}']", namespace).text
        except AttributeError as e:
            definendum = None

        try:
            full_def = "".join(
                root.find(f".//*[@defines='#{definendum_id}']").itertext())
        except Exception:
            full_def = None

        definiens = []
        references = []
        for body in definition_body_elements:
            body_text, body_refs = self._extract_body_and_references(
                body, root, namespace)
            if body_text:
                definiens.append(body_text)
                references.extend(body_refs)

        return definendum, ' '.join(definiens), references, full_def

    def append_refs(self, row):
        definition = row['joined_definition']
        references = row['references']
        dataset = row['provenance']
        def_with_refs = definition + '\n\n' + 'References: \n'
        resolved_refs = []
        for ref in references:
            res = self.existdb_handler.resolve_reference(ref, dataset)
            if res:
                resolved_refs.append(res)

        if resolved_refs:
            def_with_refs += '\n'.join(resolved_refs)
            return def_with_refs
        else:
            return definition

    @staticmethod
    def _extract_body_and_references(body, root, namespace) -> Tuple[str, List[str]]:
        """Extract text and references from definition body."""
        body_id = body.attrib.get('href', '').lstrip('#')
        body_element = root.find(
            f".//akn:defBody[@eId='{body_id}']", namespace)
        if body_element is not None:
            body_text = ''.join(body_element.itertext())
            references = [
                ref.attrib.get('href', '') for ref in body_element.findall('.//akn:ref', namespace)
            ]
            return body_text, references
        return None, None
    
    @staticmethod
    def _clean_definendum(text: str) -> str:
        text = text.strip()
        if text.startswith(("«", "'", "\"")) and text.endswith(("»", "'", "\"")):
            text = text[1:-1]
        return text.strip()

    @staticmethod
    def _clean_definiens(text: str) -> str:
        if text.startswith((':', ',')):
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

        for dataset in self.config.DATASETS:
            target = Path(self.data_dir / dataset)
            for file in target.rglob('*.xml'):
                total += 1
                try:
                    definitions = self.parse_xml(file, dataset)
                    if definitions:
                        extracted += 1
                        output_file = self.definitions_output_dir / \
                            file.name.replace('.xml', '.tsv')
                        self._save_definitions(definitions, output_file)
                except Exception as e:
                    errors += 1
                    logging.error(f"Error processing {file}: {e}")

        logging.info(f'Extracted definitions from {extracted} files. {
                     errors} errors out of {total} files.')

    def _save_definitions(self, definitions: List[Dict], output_file: Path) -> None:
        """Save extracted definitions to TSV file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df = pl.DataFrame(definitions)
        df.write_csv(output_file, separator="\t")
        logging.info(f"Saved {len(definitions)} definitions to {output_file}")


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
        output_path = Path(settings.DB_CONFIG.DEFINITIONS_OUTPUT_DIR) / "definitions.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        processed_df.write_csv(str(output_path))
        logging.info(f"Saved {len(processed_df)} definitions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract legislative definitions from a collection of AKN documents.")
    parser.add_argument('--pickle', action='store_true', default=False, 
                    help="Save extracted definitions as a pickle file for embedding later.")


    args = parser.parse_args()

    main(pickle=args.pickle)
