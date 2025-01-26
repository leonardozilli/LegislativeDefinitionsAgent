from pymilvus import connections, WeightedRanker, Collection
from langchain.embeddings.base import Embeddings
from torch.cuda import is_available as cuda_available
from milvus_model.hybrid import BGEM3EmbeddingFunction
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from functools import lru_cache

from ..settings import settings
from ..utils import setup_logging
from .utils import connect_to_milvus


class BGEMilvusDenseEmbeddings(Embeddings):
    def __init__(self):
        self.model = BGEM3EmbeddingFunction(
            model_name='BAAI/bge-m3',
            device='cuda' if cuda_available() else 'cpu',
            use_fp16=True if cuda_available() else False  # set to false if device='cpu'
        )

    def embed_documents(self, texts):
        embeddings = self.model.encode_documents(texts)
        return [i.tolist() for i in embeddings["dense"]]

    def embed_query(self, text):
        embedding = self.model.encode_queries([text])
        return embedding["dense"][0].tolist()


class BGEMilvusSparseEmbeddings(Embeddings):
    def __init__(self):
        self.model = BGEM3EmbeddingFunction(
            model_name='BAAI/bge-m3',
            device='cuda' if cuda_available() else 'cpu',
            use_fp16=True if cuda_available() else False  # set to false if device='cpu'
        )

    def embed_documents(self, texts):
        embeddings = self.model.encode_documents(texts)
        return [i.tolist() for i in embeddings["sparse"]]

    def embed_query(self, text):
        embedding = self.model.encode_queries([text])
        return embedding["sparse"]._getrow(0)


@lru_cache
def setup_retriever(k=10):
    connect_to_milvus(settings.MILVUSDB_URI)

    sparse_search_params = {"metric_type": "IP"}
    dense_search_params = {"metric_type": "COSINE", "params": {}}
    retriever = MilvusCollectionHybridSearchRetriever(
        collection=Collection(settings.MILVUSDB_COLLECTION_NAME),
        rerank=WeightedRanker(1.0, 0.7),
        anns_fields=["dense_vector", "sparse_vector"],
        field_embeddings=[
            BGEMilvusDenseEmbeddings(), BGEMilvusSparseEmbeddings()],
        field_search_params=[dense_search_params, sparse_search_params],
        top_k=k,
        text_field="definition_text",
    )

    return retriever
