from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
import logging
from typing import Any
import polars as pl
from ..config import DB_CONFIG, MILVUSDB_URI, MILVUSDB_COLLECTION_NAME

logger = logging.getLogger(__name__)

class VectorDBBuilder:
    def __init__(self, embedding_function: Any):
        self.config = DB_CONFIG
        self.milvus_uri = MILVUSDB_URI
        self.collection_name = MILVUSDB_COLLECTION_NAME
        self.batch_size = self.config['BATCH_SIZE']
        self.ef = embedding_function
        self.dense_dim = self.ef.dim["dense"]
        
    def setup_collection(self) -> Collection:
        """Setup Milvus collection with proper schema."""
        fields = [
            FieldSchema(
                name="id", 
                dtype=DataType.INT64,
                is_primary=True, 
                auto_id=True, 
                max_length=100
            ),
            FieldSchema(
                name="definition_text", 
                dtype=DataType.VARCHAR, 
                max_length=5000
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
                name="references", 
                dtype=DataType.ARRAY, 
                element_type=DataType.VARCHAR, 
                max_capacity=20
            ),
            FieldSchema(
                name="dense_vector", 
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dense_dim
            ),
        ]
        
        schema = CollectionSchema(fields, "Definitions embeddings")
        collection_name = "Definitions"
        
        # Drop existing collection if it exists
        if utility.has_collection(collection_name):
            Collection(collection_name).drop()
            
        collection = Collection(
            collection_name, 
            schema, 
            consistency_level="Strong"
        )
        
        # Create and load index
        dense_index = {
            "index_type": "AUTOINDEX", 
            "metric_type": "IP"
        }
        collection.create_index("dense_vector", dense_index)
        collection.load()
        
        return collection

    def build_vector_db(self, df: pl.DataFrame) -> None:
        """Build vector database from processed definitions."""
        try:
            # Connect to Milvus
            connections.connect(uri=self.config.milvus_url)
            
            # Setup collection
            collection = self.setup_collection()
            
            # Generate embeddings and insert in batches
            for i in range(0, len(df), self.config.batch_size):
                batch_df = df.slice(i, self.config.batch_size)
                
                # Generate embeddings for the batch
                batch_texts = batch_df['definition_text'].to_list()
                batch_embeddings = self.ef(batch_texts)
                
                # Prepare batch data
                batch_data = [
                    batch_df['definition_text'].to_list(),
                    batch_df['dataset'].to_list(),
                    batch_df['document_id'].to_list(),
                    batch_df['references'].to_list(),
                    batch_embeddings["dense"],
                ]
                
                # Insert batch
                collection.insert(batch_data)
                
            logger.info(f"Number of entities inserted: {collection.num_entities}")
            
        except Exception as e:
            logger.error(f"Error building vector database: {e}")
            raise
        finally:
            connections.disconnect(alias='default')