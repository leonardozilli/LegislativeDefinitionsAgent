from pymilvus import connections
from langchain_milvus import Milvus
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from langchain.embeddings.base import Embeddings
from torch.cuda import is_available as cuda_available

from functools import lru_cache

from LegalDefAgent.src.config import MILVUSDB_URI, MILVUSDB_COLLECTION_NAME
from LegalDefAgent.src.utils import setup_logging


@lru_cache
def setup_vectorstore(milvusdb_uri=None):
    #logger.info(f"Setting up vector store with URI: {milvusdb_uri}")
    #import traceback
    #logger.info("Vector store setup called from:")
    #for line in traceback.format_stack():
        #logger.info(line.strip())

    class BGEMilvusEmbeddings(Embeddings):
        def __init__(self):
            self.model = BGEM3EmbeddingFunction(
                model_name='BAAI/bge-m3',
                device='cuda' if cuda_available() else 'cpu',
                use_fp16=True if cuda_available() else False #set to false if device='cpu'
            )

        def embed_documents(self, texts):
            embeddings = self.model.encode_documents(texts)
            return [i.tolist() for i in embeddings["dense"]]

        def embed_query(self, text):
            embedding = self.model.encode_queries([text])
            return embedding["dense"][0].tolist()

    vectorstore = Milvus(
        embedding_function=BGEMilvusEmbeddings(),
        connection_args={"uri": MILVUSDB_URI if milvusdb_uri is None else milvusdb_uri},
        collection_name=MILVUSDB_COLLECTION_NAME,
        vector_field="dense_vector",
        text_field="definition_text",
    )


    return vectorstore