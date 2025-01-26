from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection, MilvusClient
import argparse
import logging
import pickle
import polars as pl
from pathlib import Path
from milvus_model.hybrid import BGEM3EmbeddingFunction
from torch.cuda import is_available as cuda_available

from LegalDefAgent.src.utils import setup_logging
from LegalDefAgent.src.settings import settings

setup_logging()


def setup_collection(collection_name, dense_dim) -> Collection:
    """Setup Milvus collection with proper schema."""
    fields = [
        FieldSchema(
            name="id", 
            dtype=DataType.INT64,
            is_primary=True, 
            #auto_id=True, 
            #max_length=100
        ),
        FieldSchema(
            name="definition_text", 
            dtype=DataType.VARCHAR, 
            max_length=5000
        ),
        FieldSchema(
            name="definendum_label", 
            dtype=DataType.VARCHAR, 
            max_length=256
        ),
        FieldSchema(
            name="dataset", 
            dtype=DataType.VARCHAR, 
            max_length=10
        ),
        FieldSchema(
            name="document_id", 
            dtype=DataType.VARCHAR, 
            max_length=40
        ),
        FieldSchema(
            name="frbr_work", 
            dtype=DataType.VARCHAR, 
            max_length=120
        ),
        FieldSchema(
            name="frbr_expression", 
            dtype=DataType.VARCHAR, 
            max_length=120
        ),
        FieldSchema(
            name="sparse_vector", 
            dtype=DataType.SPARSE_FLOAT_VECTOR,
        ),
        FieldSchema(
            name="dense_vector", 
            dtype=DataType.FLOAT_VECTOR,
            dim=dense_dim
        ),
    ]


    schema = CollectionSchema(fields, "Definitions embeddings")
    
    # Drop existing collection if it exists
    if utility.has_collection(collection_name):
        Collection(collection_name).drop()
        
    collection = Collection(
        collection_name, 
        schema, 
        consistency_level="Strong"
    )
    
    # Create and load index
    sparse_index = {
        "index_type": "SPARSE_INVERTED_INDEX", 
        "metric_type": "IP"
    }
    dense_index = {
        "index_type": "FLAT", 
        "metric_type": "COSINE"
    }
    collection.create_index("sparse_vector", sparse_index)
    collection.create_index("dense_vector", dense_index)
    collection.load()
    
    return collection


def populate_vectorstore(definitions_corpus, definitions_embeddings, settings):
    """Build vector database from processed definitions."""

    defs_df = pl.read_csv(definitions_corpus)
    defs_embeddings = pickle.load(open(definitions_embeddings, 'rb'))

    ef = BGEM3EmbeddingFunction(
                model_name='BAAI/bge-m3',
                device='cuda' if cuda_available() else 'cpu',
                use_fp16=True if cuda_available() else False #set to false if device='cpu'
            )

    logging.info("Building vector database...")
    try:
        # create parent directory if it doesn't exist
        Path(settings.MILVUSDB_URI).parent.mkdir(parents=True, exist_ok=True)
        client = MilvusClient(
                uri=settings.MILVUSDB_URI
            )
        connections.connect(uri=settings.MILVUSDB_URI)
        
        # Setup collection
        collection = setup_collection(settings.MILVUSDB_COLLECTION_NAME, ef.dim["dense"])

        batch_size = 50
        
        # Insert embeddings in batches
        for i in range(0, len(defs_df), batch_size):
            batch_df = defs_df.slice(i, batch_size)
            
            batch_sparse_embeddings = defs_embeddings['sparse'][i:i+batch_size]
            batch_dense_embeddings = defs_embeddings['dense'][i:i+batch_size]
            
            batch_data = [
                batch_df['id'].to_list(),
                batch_df['definition_text'].to_list(),
                batch_df['label'].to_list(),
                batch_df['dataset'].to_list(),
                batch_df['document_id'].to_list(),
                batch_df['frbr_work'].to_list(),
                batch_df['frbr_expression'].to_list(),
                batch_sparse_embeddings,
                batch_dense_embeddings
            ]
            
            # Insert batch
            collection.insert(batch_data)
            
        logging.info(f"Inserted {collection.num_entities} entities into vector database located at {settings.MILVUSDB_URI}")
        
    except Exception as e:
        logging.error(f"Error building vector database: {e}")
        raise
    finally:
        connections.disconnect(alias='default')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initiate and populate the vector database of legal definitions.")
    parser.add_argument('--defs_corpus', '-c', type=str, required=True, help='Path to the corpus definitions to insert into the database.')
    parser.add_argument('--defs_embeddings', '-e', type=str, required=True, help='Path to the embeddings of the definitions.')
    #parser.add_argument('--datasets', nargs='+', help='Optional list of datasets to process. If None, uses all configured datasets.')
    args = parser.parse_args()

    populate_vectorstore(definitions_corpus=args.defs_corpus, definitions_embeddings=args.defs_embeddings, settings=settings)