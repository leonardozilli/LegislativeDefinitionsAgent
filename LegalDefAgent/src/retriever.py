from pymilvus import connections
from langchain_milvus import Milvus
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from langchain.embeddings.base import Embeddings
from torch.cuda import is_available as cuda_available

import src.config as config

def setup_vectorstore(k: int = 10):
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
        connection_args={"uri": config.MILVUSDB_URI},#MILVUS_URI},
        collection_name=config.MILVUSDB_COLLECTION_NAME,
        vector_field="dense_vector",
        text_field="definition_text",
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return retriever


def drop_all_connections():
    """
    Drops all connections to the Milvus database.
    """
    for alias, conn in connections.list_connections():
        connections.remove_connection(alias)