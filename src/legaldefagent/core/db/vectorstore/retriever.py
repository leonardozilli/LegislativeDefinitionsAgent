from functools import lru_cache

from langchain.embeddings.base import Embeddings
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from pymilvus import Collection, WeightedRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from torch.cuda import is_available as cuda_available

from legaldefagent.settings import settings
from legaldefagent.core.db.vectorstore.utils import connect_to_milvus


class BGEMilvusDenseEmbeddings(Embeddings):
    def __init__(self):
        self.model = BGEM3EmbeddingFunction(
            model_name="BAAI/bge-m3",
            device="cuda" if cuda_available() else "cpu",
            use_fp16=True if cuda_available() else False,
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
            model_name="BAAI/bge-m3",
            device="cuda" if cuda_available() else "cpu",
            use_fp16=True if cuda_available() else False,
        )

    def embed_documents(self, texts):
        embeddings = self.model.encode_documents(texts)
        return [i.tolist() for i in embeddings["sparse"]]

    def embed_query(self, text):
        embedding = self.model.encode_queries([text])
        return embedding["sparse"]._getrow(0)


class CustomMilvusHybridSearchRetriever(MilvusCollectionHybridSearchRetriever):
    def _process_search_result(self, search_results):
        documents = []
        for result in search_results[0]:
            # if result.distance > 0.8:
            if self.output_fields:
                data = {field: result.entity.get(field) for field in self.output_fields}
                doc = self._parse_document(data)
                documents.append(doc)
        return documents


@lru_cache(maxsize=1)
def setup_retriever(k=10):
    connect_to_milvus(settings.milvusdb.path)

    sparse_search_params = {"metric_type": "IP"}
    dense_search_params = {"metric_type": "COSINE", "params": {}}

    retriever = CustomMilvusHybridSearchRetriever(
        collection=Collection(settings.milvusdb.collection_name),
        rerank=WeightedRanker(1.0, 0.7),
        anns_fields=["dense_vector", "sparse_vector"],
        field_embeddings=[BGEMilvusDenseEmbeddings(), BGEMilvusSparseEmbeddings()],
        field_search_params=[dense_search_params, sparse_search_params],
        top_k=k,
        text_field="definition_text",
    )

    return retriever
