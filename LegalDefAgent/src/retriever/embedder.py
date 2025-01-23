from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection, MilvusClient
import logging
from typing import Any
import polars as pl
from pathlib import Path
from milvus_model.hybrid import BGEM3EmbeddingFunction
from ..settings import settings
from torch.cuda import is_available as cuda_available

logger = logging.getLogger(__name__)

class VectorDBBuilder:
    def __init__(self):
        self.config = settings
        self.milvus_uri = self.config.MILVUSDB_URI
        self.collection_name = self.config.MILVUSDB_COLLECTION_NAME
        self.batch_size = self.config.DB_CONFIG.BATCH_SIZE
        self.ef = BGEM3EmbeddingFunction(
                model_name='BAAI/bge-m3',
                device='cuda' if cuda_available() else 'cpu',
                use_fp16=True if cuda_available() else False #set to false if device='cpu'
            )
        self.dense_dim = self.ef.dim["dense"]
        
    def setup_collection(self) -> Collection:
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
    
    def build_vector_db(self, df: pl.DataFrame, defs_embeddings=None) -> None:
        """Build vector database from processed definitions."""
        logger.info("Building vector database...")
        try:

            Path(self.milvus_uri).parent.mkdir(parents=True, exist_ok=True)
            client = MilvusClient(
                uri=self.milvus_uri
            )
            connections.connect(uri=self.milvus_uri)
            
            # Setup collection
            collection = self.setup_collection()
            
            # Generate embeddings and insert in batches
            for i in range(0, len(df), self.batch_size):
                batch_df = df.slice(i, self.batch_size)
                
                # Generate embeddings for the batch
                batch_texts = batch_df['definition_text'].to_list()
                if not defs_embeddings:
                    batch_embeddings = self.ef(batch_texts)
                else:
                    batch_sparse_embeddings = defs_embeddings['sparse'][i:i+self.batch_size]
                    batch_dense_embeddings = defs_embeddings['dense'][i:i+self.batch_size]
                
                # Prepare batch data
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
                
            logger.info(f"Inserted {collection.num_entities} entities into vector database")
            
        except Exception as e:
            logger.error(f"Error building vector database: {e}")
            raise
        finally:
            connections.disconnect(alias='default')