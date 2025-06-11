import logging
import pickle
from pathlib import Path

import polars as pl
import typer
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

logger = logging.getLogger(__name__)

app = typer.Typer()


def embed_definitions(
    defs_path: Path = typer.Option(..., "--defs", "-i", help="Path to the CSV file containing legal definitions."),
    output_dir: Path = typer.Option(
        Path("data/embeddings/"), "--output_dir", "-o", help="Directory to save the embeddings files in."
    ),
    target_definiendums: bool = typer.Option(
        False,
        "--definiendums",
        "-d",
        help="Whether to generate embeddings for just definiendums instead of the whole definitions.",
    ),
    embedding_type: str = typer.Option(
        "hybrid",
        "--type",
        "-t",
        help="Type of embedding to generate, one of 'dense', 'sparse', or 'hybrid'.",
        show_choices=True,
    ),
):
    defs = pl.read_csv(defs_path)

    if embedding_type == "dense":
        ef = BGEM3EmbeddingFunction(use_fp16=False, return_sparse=False)
    elif target_definiendums or embedding_type == "sparse":
        ef = BGEM3EmbeddingFunction(use_fp16=False, return_dense=False)
    elif embedding_type == "hybrid":
        ef = BGEM3EmbeddingFunction(use_fp16=False)

    logger.debug(f"Using {embedding_type} embedding function")

    if target_definiendums:
        logger.info("Extracting definiendums only")
        docs = (
            defs.select(
                pl.col("label")
                .str.replace("#", "")
                .str.replace(r"([a-zà-ÿ])([A-Z])", r"${1} ${2}", n=-1)  # reverse camel case
                .str.to_lowercase()
            )
        )["label"].to_list()
    else:
        docs = defs["definition_text"].to_list()

    embeddings = ef(docs)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{'defs' if not target_definiendums else 'definiendums'}_embeddings_{embedding_type}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(embeddings, f)

    logger.info(f"{'Definitions' if not target_definiendums else 'Definiendums'} embeddings saved to {out_file}")


app.command()(embed_definitions)

if __name__ == "__main__":
    app()
