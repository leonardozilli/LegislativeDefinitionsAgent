import tiktoken
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from pymilvus import connections
from IPython.display import Image, display
from langchain_milvus import Milvus
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from langchain.embeddings.base import Embeddings
from torch.cuda import is_available as cuda_available

load_dotenv(find_dotenv())


def get_token_count(string: str, llm_name: str) -> int:
    """
    Returns the number of tokens in a given string for a specified language model.
    :param string: The input text to be tokenized.
    :param llm_name: The name of the language model to determine the encoding.
    :return: The count of tokens in the input string.
    """
    encoding = tiktoken.encoding_for_model(llm_name)
    return len(encoding.encode(string))


def chatqa(query: str, llm_name: str) -> str:
    """
    Basic OpenAI inference through langchain.
    :param query: The user's query.
    :param llm_name: The name of the OpenAI model to be used.
    :return: The model's output
    """

    llm = ChatOpenAI(model_name=llm_name, temperature=0)
    chain = LLMChain(llm=llm, prompt=PromptTemplate(template="""{q}""", input_variables=["q"]))
    res = chain.apply([{'q': query}])[0]['text']

    return res


def merge_dicts(*dicts: dict) -> dict:
    _ = {}
    for dict_ in dicts:
        _.update(dict_)
    return _


def drop_all_connections():
    """
    Drops all connections to the Milvus database.
    """
    for alias, conn in connections.list_connections():
        connections.remove_connection(alias)
 

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

    MILVUS_URI = "../vec_db/definitions_vectors.db"
    MILVUS_COLLECTION_NAME = 'Definitions'

    vectorstore = Milvus(
        embedding_function=BGEMilvusEmbeddings(),
        connection_args={"uri": MILVUS_URI},
        collection_name=MILVUS_COLLECTION_NAME,
        vector_field="dense_vector",
        text_field="definition_text",
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return retriever


def doc_to_json(doc):
    return {'metadata': doc.metadata, 'definition': doc.page_content}


def draw_graph(graph):
    try:
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        pass