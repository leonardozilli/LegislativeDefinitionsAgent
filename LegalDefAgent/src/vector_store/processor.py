import polars as pl
from pathlib import Path
from typing import List
from ..config import DB_CONFIG

class DefinitionProcessor:
    def __init__(self):
        self.config = DB_CONFIG
        self.max_length = self.config['MAX_DEFINITION_LENGTH']
        
    def process_definitions(self, tsv_files: List[Path]) -> pl.DataFrame:
        """Process and transform definition data from TSV files."""
        # Read all TSV files into a single DataFrame
        df = pl.concat([
            pl.read_csv(f, separator='\t') 
            for f in tsv_files
        ])
        
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
            .filter(pl.col('full_definition').str.len_chars() < self.config.max_definition_length)
            .select(
                pl.col('joined_definition').alias('definition_text'),
                pl.col('provenance').alias('dataset'),
                pl.col('document').alias('document_id'),
                pl.col('references'),
            )
            .with_columns([
                pl.col('references').map_elements(eval, return_dtype=pl.List(pl.String))
            ])
            .with_row_index('id')
        )
        
        return processed_df