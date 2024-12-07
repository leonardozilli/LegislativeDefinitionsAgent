import tiktoken
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from IPython.display import Image, display

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
    chain = LLMChain(llm=llm, prompt=PromptTemplate(
        template="""{q}""", input_variables=["q"]))
    res = chain.apply([{'q': query}])[0]['text']

    return res


def merge_dicts(*dicts: dict) -> dict:
    _ = {}
    for dict_ in dicts:
        _.update(dict_)
    return _


def docs_to_json(docs):
    return {doc.to_json().get('kwargs') for doc in docs}


def json_to_aimessage(parsed_output):
    """Convert parsed output back to AIMessage."""
    content = f"{parsed_output}"
    return AIMessage(content=content)


def draw_graph(graph):
    try:
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        pass


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    node = event
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


#utils
import re
import csv
import os
from lxml import etree

AKN_NAMESPACE = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
AKN_PREFIX = "{" + AKN_NAMESPACE + "}"
AKN_PREFIX_XPATH = {"akomaNtoso": AKN_NAMESPACE}




frbr_map = {}
file_map = {}

frbr_map["it"] = {}
file_map["it"] = {}

def res():
    with open("./metadata/filename_mapping_normattiva.csv") as f:
        reader = csv.DictReader(f)
        for line in reader:
            frbrwork = line["FRBRWork"]
            frbrshort = line["FRBRShort"]
            frbrexpression = line["FRBRExpression"]
            filename = line["filename"]
            if frbrwork not in frbr_map["it"]:
                frbr_map["it"][frbrwork] = []
            frbr_map["it"][frbrwork].append(frbrexpression)
            if frbrshort not in frbr_map["it"]:
                frbr_map["it"][frbrshort] = []
            frbr_map["it"][frbrshort].append(frbrexpression)

            file_map["it"][frbrexpression] = filename


    frbr_map["en"] = {}
    file_map["en"] = {}


    with open("./metadata/filename_mapping_leos.csv") as f:
        reader = csv.DictReader(f)
        for line in reader:
            frbrwork = line["FRBRWork"]
            frbrshort = line["FRBRShort"]
            frbrexpression = line["FRBRExpression"]
            filename = line["filename"]
            if frbrwork not in frbr_map["en"]:
                frbr_map["en"][frbrwork] = []
            frbr_map["en"][frbrwork].append(frbrexpression)
            if frbrshort not in frbr_map["en"]:
                frbr_map["en"][frbrshort] = []
            frbr_map["en"][frbrshort].append(frbrexpression)

            file_map["en"][frbrexpression] = filename

    for lang in ["it", "en"]:
        for expressions in frbr_map[lang].values():
            expressions.sort()


def get_uri_date(uri):
    at_split = uri.split("@")[-1]
    date = at_split.split("/")[0]
    return date


def resolve_reference(uri, references_dir, date=None, lang="it"):
    if lang == "it":
        uri = uri.lower()
        title_name = "docTitle"
    elif lang == "en":
        uri = re.sub(r"/ep/", "/", uri)
        title_name = "longTitle"

    else:
        raise NotImplementedError

    if "@" in uri:
        date = get_uri_date(uri)
        uri = uri.split("@")[-1]
        uri = "/".join(uri.split("/")[:-1])
    print(f"resolving {uri}")
    if "#" in uri and lang == "it":
        uri, qualifiers = uri.split("#")
        qualifiers = re.sub("-", "__", qualifiers)
    elif "~" in uri and lang == "en":
        uri, qualifiers = uri.split("~")
    else:
        qualifiers = ""

    if uri not in frbr_map[lang]:
        print(f"could not resolve reference {uri}")
        return
    frbrexpressions = frbr_map[lang][uri]

    # select work based on date, for now i get the last
    if date is None:
        frbrexpression = frbrexpressions[-1]
    else:
        frbrexpression = None
        for curr_expression in frbrexpressions:
            if curr_expression < date:
                frbrexpression = curr_expression

    file_name = file_map[lang][frbrexpression]
    file_path = os.path.join(references_dir, file_name)
    print(f"found ref file {file_name}")
    file_tree = etree.parse(file_path)
    if qualifiers == "":
        title = file_tree.xpath(
            f".//akomaNtoso:{title_name}", namespaces=AKN_PREFIX_XPATH
        )[0]
        articles = file_tree.xpath(
            ".//akomaNtoso:article", namespaces=AKN_PREFIX_XPATH
        )
        print("resolved")
        return title, articles[0]
    else:
        qualified_elements = file_tree.xpath(f".//*[@eId='{qualifiers}']")
        if qualified_elements is None or not len(qualified_elements):
            print("could not resolve, uri mistake")
            return
        print("resolved")
        return qualified_elements[0]
