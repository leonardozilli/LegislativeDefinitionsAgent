import json
import logging
import pickle
import re
import xml.etree.ElementTree as ET
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
import typer
from tqdm import tqdm

from legaldefagent.existdb import existdb_handler
from legaldefagent.settings import settings

logger = logging.getLogger(__name__)

app = typer.Typer()


class DefinitionExtractor:
    def __init__(self):
        self.collections = settings.COLLECTIONS
        self.existdb_handler = existdb_handler

    def extract_from_local(self, input_dir: Path = None) -> pl.DataFrame:
        """Extract definitions from local datasets into a DataFrame."""
        all_definitions = []
        extracted, errors, total = 0, 0, 0

        for dataset in self.collections:
            target = input_dir / dataset
            logger.info(f"Extracting {dataset} definitions from {target}...")
            files = target.rglob("*.xml")
            for file in tqdm(list(files)):
                total += 1
                try:
                    data = self.parse_xml(file, dataset)
                    if data:
                        extracted += 1
                        all_definitions.extend(data)
                except Exception as e:
                    errors += 1
                    logger.error(f"Error processing {file}: {e}")

        logger.info(f"Extracted definitions from {extracted} files. {errors} errors out of {total} files.")

        if not all_definitions:
            raise ValueError("No definitions were extracted")

        final_df = self.filter_definitions(pl.DataFrame(all_definitions))

        return final_df

    def extract_from_exist(self) -> pl.DataFrame:
        """Extract definitions from eXistDB into a DataFrame."""
        all_definitions = []

        for dataset in self.collections:
            logger.info(f"Extracting {dataset} definitions from eXistDB...")
            raw_result = self.existdb_handler.get_all_definitions(dataset)
            defs_list = json.loads(raw_result)["definition"]
            if not defs_list:
                logger.warning(f"No definitions found for dataset {dataset} in eXistDB.")
                continue
            all_definitions.extend(defs_list)

        if not all_definitions:
            raise ValueError("No definitions were extracted")

        defs_df = pl.DataFrame(all_definitions).with_columns(
            pl.col("references").str.split(", ").cast(pl.List(pl.Utf8)).fill_null([]),
            pl.col("keywords").fill_null(""),
        )
        final_df = self.filter_definitions(defs_df)

        return final_df

    def filter_definitions(self, df: pl.DataFrame) -> pl.DataFrame:
        filtered_df = (
            df.with_columns(
                pl.when(pl.col("definiendum").is_null() | pl.col("definiens").is_null())
                .then(pl.col("full_definition"))
                .otherwise(pl.concat_str([pl.col("definiendum"), pl.col("definiens")], separator=": "))
                .alias("joined_definition")
            )
            .drop_nulls("joined_definition")
            .with_columns(
                pl.struct(
                    pl.col("joined_definition"),
                    pl.col("references"),
                    pl.col("provenance"),
                )
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

        return filtered_df

    def parse_xml(self, xml_file: Path, dataset: str) -> Optional[List[Dict]]:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            namespace = self.collections[dataset]["namespace"]

            frbr_work = (
                root.find(".//akn:FRBRWork", namespace).find(".//akn:FRBRthis", namespace).attrib.get("value", "")
            )
            frbr_expression = (
                root.find(".//akn:FRBRExpression", namespace).find(".//akn:FRBRthis", namespace).attrib.get("value", "")
            )
            keywords = root.findall(".//akn:keyword", namespace)

            definitions_el = root.findall(".//akn:definitions", namespace)
            if not definitions_el:
                return None

            definitions = []
            for definition in root.findall(".//akn:definition", namespace):
                try:
                    definiendum, definiens, references, full_def = self._parse_definition(definition, root, namespace)
                    if definiendum:
                        definitions.append(
                            {
                                "def_n": definition.find(".//akn:definitionHead", namespace).attrib.get("href", ""),
                                "label": definition.find(".//akn:definitionHead", namespace).attrib.get("refersTo", ""),
                                "definiendum": self._clean_definiendum(definiendum),
                                "definiens": self._clean_definiens(definiens),
                                "full_definition": self._clean_full_def(full_def),
                                "references": references,
                                "provenance": dataset,
                                "document": xml_file.name,
                                "frbr_work": frbr_work,
                                "frbr_expression": frbr_expression,
                                "keywords": ", ".join([kw.attrib.get("showAs", "") for kw in keywords]),
                            }
                        )
                except Exception as e:
                    logger.error(f"Error parsing definition in {xml_file}: {e}")
                    pass

            return definitions
        except Exception as e:
            logger.error(f"Error processing file {xml_file}: {e}")
            return None

    def _parse_definition(self, definition, root, namespace) -> Tuple[str, str, List[str], str]:
        """Extract components from a single definition."""
        definition_head = definition.find(".//akn:definitionHead", namespace)
        definition_body_elements = definition.findall(".//akn:definitionBody", namespace)

        definiendum_id = definition_head.attrib.get("href", "").lstrip("#")
        try:
            definiendum = root.find(f".//akn:def[@eId='{definiendum_id}']", namespace).text
        except AttributeError:
            definiendum = None

        try:
            full_def = "".join(root.find(f".//*[@defines='#{definiendum_id}']").itertext())
        except Exception:
            full_def = None

        definiens = []
        references = []
        for body in definition_body_elements:
            body_text, body_refs = self._extract_body_and_references(body, root, namespace)
            if body_text:
                definiens.append(body_text)
                references.extend(body_refs)

        return definiendum, " ".join(definiens), references, full_def

    def append_refs(self, row):
        definition = row["joined_definition"]
        references = row["references"]
        dataset = row["provenance"]
        def_with_refs = definition + "\n\n" + "References: \n"
        resolved_refs = []
        for ref in references:
            res = self.existdb_handler.resolve_reference(ref, dataset)
            if res:
                resolved_refs.append(res)

        if resolved_refs:
            def_with_refs += "\n".join(resolved_refs)
            return def_with_refs
        else:
            return definition

    @staticmethod
    def _extract_body_and_references(body, root, namespace) -> Tuple[str, List[str]]:
        """Extract text and references from definition body."""
        body_id = body.attrib.get("href", "").lstrip("#")
        body_element = root.find(f".//akn:defBody[@eId='{body_id}']", namespace)
        if body_element is not None:
            body_text = "".join(body_element.itertext())
            references = [ref.attrib.get("href", "") for ref in body_element.findall(".//akn:ref", namespace)]
            return body_text, references
        return None, None

    @staticmethod
    def _clean_definiendum(text: str) -> str:
        text = text.strip()
        if text.startswith(("«", "'", '"')) and text.endswith(("»", "'", '"')):
            text = text[1:-1]
        return text.strip()

    @staticmethod
    def _clean_definiens(text: str) -> str:
        if text.startswith((":", ",")):
            text = text.lstrip(":,")
        return re.sub(r"\s+", " ", text.strip().replace("\n", " "))

    @staticmethod
    def _clean_full_def(text: str) -> str:
        if text is None:
            return ""
        return re.sub(r"\s+", " ", text.strip().replace("\n", " ").replace('""', ""))

    def extract_all(self) -> None:
        """Extract definitions from all datasets."""
        extracted, errors, total = 0, 0, 0
        self.definitions_output_dir.mkdir(parents=True, exist_ok=True)

        for dataset in self.config.DATASETS:
            target = Path(self.data_dir / dataset)
            for file in target.rglob("*.xml"):
                total += 1
                try:
                    definitions = self.parse_xml(file, dataset)
                    if definitions:
                        extracted += 1
                        output_file = self.definitions_output_dir / file.name.replace(".xml", ".tsv")
                        self._save_definitions(definitions, output_file)
                except Exception as e:
                    errors += 1
                    logger.error(f"Error processing {file}: {e}")

        logger.info(f"Extracted definitions from {extracted} files. {errors} errors out of {total} files.")

    def _save_definitions(self, definitions: List[Dict], output_file: Path) -> None:
        """Save extracted definitions to TSV file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df = pl.DataFrame(definitions)
        df.write_csv(output_file, separator="\t")
        logger.info(f"Saved {len(definitions)} definitions to {output_file}")


def save_definitions_list_to_pickle(df, output_file="definitions_list.pkl"):
    logger.info("Saving definitions list to pickle file...")
    definitions_list = df["definition_text"].to_list()
    with open(output_file, "wb") as f:
        pickle.dump(definitions_list, f)
    logger.info(f"Definitions list saved to {output_file}")


class Source(str, Enum):
    local = "local"
    exist = "exist"


def extract_definitions(
    pickle: bool = typer.Option(False, help="Save extracted definitions as a pickle file for embedding later."),
    source: Source = typer.Option(..., "--source", "-s", help="Source of definitions: 'local' or 'exist'"),
    input_dir: Path = typer.Option(
        None,
        "--input-dir",
        "-i",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Input directory for local XML collections",
    ),
):
    if source == Source.local and input_dir is None:
        raise typer.BadParameter("You must provide --input-dir when using --source local.")

    extractor = DefinitionExtractor()

    if source == Source.local:
        definitions_df = extractor.extract_from_local(input_dir)
    elif source == Source.exist:
        definitions_df = extractor.extract_from_exist()
    else:
        raise ValueError("Invalid source. Choose 'local' or 'exist'.")

    if pickle:
        save_definitions_list_to_pickle(definitions_df)
    else:
        output_path = Path("data/definitions_corpus") / "definitions.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        definitions_df.write_csv(str(output_path))
        logger.info(f"Saved {len(definitions_df)} definitions to {output_path}")


app.command()(extract_definitions)

if __name__ == "__main__":
    app()
