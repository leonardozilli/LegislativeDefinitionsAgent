import logging
import pickle
from pathlib import Path

import polars as pl
import typer
from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      MilvusClient, connections, utility)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from torch.cuda import is_available as cuda_available

logger = logging.getLogger(__name__)

app = typer.Typer()


def setup_collection(collection_name, dense_dim) -> Collection:
    """Setup Milvus collection with proper schema."""
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
        ),
        FieldSchema(name="definition_text", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="definiendum_label", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="dataset", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=40),
        FieldSchema(name="frbr_work", dtype=DataType.VARCHAR, max_length=120),
        FieldSchema(name="frbr_expression", dtype=DataType.VARCHAR, max_length=120),
        FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=325),
        FieldSchema(
            name="sparse_vector",
            dtype=DataType.SPARSE_FLOAT_VECTOR,
        ),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
    ]

    schema = CollectionSchema(fields, "Definitions embeddings")

    if utility.has_collection(collection_name):
        Collection(collection_name).drop()

    collection = Collection(collection_name, schema, consistency_level="Strong")

    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    dense_index = {"index_type": "FLAT", "metric_type": "COSINE"}
    collection.create_index("sparse_vector", sparse_index)
    collection.create_index("dense_vector", dense_index)
    collection.load()

    return collection


def populate_vectorstore(
    definitions: Path = typer.Option(
        ...,
        "--definitions",
        "-d",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to the definitions corpus CSV.",
    ),
    embeddings: Path = typer.Option(
        ..., "--embeddings", "-e", exists=True, file_okay=True, dir_okay=False, help="Path to the pickled embeddings."
    ),
    vectorstore: Path = typer.Option(
        None,
        "--vectorstore",
        "-v",
        exists=False,
        file_okay=True,
        dir_okay=False,
        help="Path to an existing local vectorstore. If not provided, a new Milvus instance will be created at data/definitions.db",
    ),
):
    defs_df = pl.read_csv(definitions)

    with open(embeddings, "rb") as f:
        defs_embeddings = pickle.load(f)

    ef = BGEM3EmbeddingFunction(
        model_name="BAAI/bge-m3",
        device="cuda" if cuda_available() else "cpu",
        use_fp16=True if cuda_available() else False,
    )

    if not vectorstore:
        try:
            logger.info("Instantiating vector database...")
            Path("data/vdb/definitions.db").parent.mkdir(parents=True, exist_ok=True)
            client = MilvusClient(uri="data/vdb/definitions.db")  # noqa: F841
            connections.connect(uri="data/vdb/definitions.db")

            collection = setup_collection("Definitions", ef.dim["dense"])

        except Exception as e:
            logger.error(f"Error instantiating vector database: {e}")
            raise
    else:
        try:
            logger.info(f"Connecting to local vectorstore at {vectorstore}")
            connections.connect(uri=f"file://{vectorstore}")

            collection = setup_collection("Definitions", ef.dim["dense"])
            logger.info(f"Local vectorstore created at {vectorstore}")
        except Exception as e:
            logger.error(f"Error connecting to local vectorstore: {e}")
            raise

    try:
        batch_size = 50

        for i in range(0, len(defs_df), batch_size):
            batch_df = defs_df.slice(i, batch_size)

            batch_sparse_embeddings = defs_embeddings["sparse"][i : i + batch_size]
            batch_dense_embeddings = defs_embeddings["dense"][i : i + batch_size]

            batch_data = [
                batch_df["id"].to_list(),
                batch_df["definition_text"].to_list(),
                batch_df["label"].to_list(),
                batch_df["dataset"].to_list(),
                batch_df["document_id"].to_list(),
                batch_df["frbr_work"].to_list(),
                batch_df["frbr_expression"].to_list(),
                batch_df["keywords"].to_list(),
                batch_sparse_embeddings,
                batch_dense_embeddings,
            ]

            collection.insert(batch_data)

        logger.info(f"Inserted {collection.num_entities} entities into vector database.")
    except Exception as e:
        logger.error(f"Error inserting into vector database: {e}")
        raise
    finally:
        connections.disconnect(alias="default")


app.command()(populate_vectorstore)

if __name__ == "__main__":
    app()
