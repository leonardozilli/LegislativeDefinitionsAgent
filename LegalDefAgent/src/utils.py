import tiktoken
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from IPython.display import Image, display
import logging
import sys
import os
import re
from datetime import datetime, date
from .settings import settings
import xml.etree.ElementTree as ET
from pathlib import Path


load_dotenv(find_dotenv())

from typing import List, Optional, Any
from pyexistdb import db


class ExistDBHandler:
    """Handler for executing XQueries against an eXist-db instance."""
    
    def __init__(self, settings: dict):
        """
        Initialize connection to eXist-db.
        
        Args:
            server_url: Full URL to eXist-db server
            username: eXist-db username
            password: eXist-db password
        """
        self.settings = settings
        self.db = db.ExistDB(
            server_url=f"http://{self.settings.XDB_HOST}:{self.settings.XDB_PORT}/exist/",
            username=self.settings.XDB_USER,
            password=self.settings.XDB_PASSWORD)
        self.namespaces = settings.NAMESPACES
        self.collection_names_map = self.settings.COLLECTION_NAMES_MAP
    
    def _execute_query(self, query: str) -> List[str]:
        """
        Execute an XQuery and return all results.
        
        Args:
            query: XQuery string to execute
            
        Returns:
            List of results as strings
            
        Raises:
            Exception: If query execution fails
        """
        try:
            results = []
            query_result = self.db.executeQuery(query)
            hits = self.db.getHits(query_result)
            
            for i in range(hits):
                result = self.db.retrieve(query_result, i)
                results.append(result)
                
            return results
            
        except Exception as e:
            raise Exception(f"Query execution failed: {str(e)}")
    

    def get_work_eurovocs(self, def_metadata: dict) -> list:
        frbr_work = def_metadata['frbr_work']
        dataset = self.collection_names_map[def_metadata['dataset']]
        definendum_label = def_metadata['definendum_label']
        QUERY = r"""
            xquery version "3.1";
            declare namespace akn = "{namespace}";
                    
            let $work:="{frbr_work}"

            let $doc := collection('/db/{dataset}')[.//akn:FRBRWork/akn:FRBRthis/@value=$work]


            return $doc/*//akn:classification/akn:keyword/@showAs/string()
        """

        namespace = self.namespaces[dataset]['akn']
        query = QUERY.format(
            namespace=namespace, dataset=dataset, frbr_work=frbr_work, definendum_label=definendum_label)

        execute_query = self._execute_query(query)
        results = [result.data.decode() for result in execute_query]
        
        return results
    
    def extract_definition_from_exist(self, def_metadata: dict) -> str:
        frbr_work = def_metadata['frbr_work']
        dataset = self.collection_names_map[def_metadata['dataset']]
        definendum_label = def_metadata['definendum_label']
        if def_metadata['dataset'] in ['EurLex', 'PDL']:
            ARTICLE_REF_QUERY = r"""
                xquery version "3.1";
                declare namespace akn = "{namespace}";
                
                let $work:="{frbr_work}"

                let $docs := collection('/db/{dataset}')[.//akn:FRBRWork/akn:FRBRthis/@value=$work]

                let $results := 
                    for $doc in $docs
                    let $expdate := $doc//akn:FRBRExpression/akn:FRBRdate/@date/string()
                    let $def := $doc//akn:definitionHead[@refersTo="{definendum_label}"]/@href/string()
                    return
                        <result>
                            <date>{{$expdate}}</date>
                        {{for $definition in $doc//*[@defines=$def]
                            return <definition>{{$definition/string()}}</definition>}}
                        </result>

                return
                    <results>{{$results}}</results>
            """
        elif def_metadata['dataset'] == 'Normattiva':
            ARTICLE_REF_QUERY = r"""
                xquery version "3.1";
                declare namespace akn = "{namespace}";
                
                let $work:="{frbr_work}"

                let $docs := collection('/db/{dataset}')[.//akn:FRBRWork/akn:FRBRthis/@value=$work]

                let $results := 
                    for $doc in $docs
                    let $expdate := $doc//akn:FRBRExpression/akn:FRBRdate/@date/string()
                    let $def := $doc//akn:definitionHead[@refersTo="{definendum_label}"]/@href/string()
                    return
                        <result>
                            <date>{{$expdate}}</date>
                        {{for $definition in $doc//*[@defines=$def]
                            return <definition>{{$definition/string()}}</definition>}}
                        </result>

                return
                    <results>{{$results}}</results>
            """


        namespace = self.namespaces[dataset]['akn']
        query = ARTICLE_REF_QUERY.format(
            namespace=namespace, dataset=dataset, frbr_work=frbr_work, definendum_label=definendum_label)

        execute_query = self._execute_query(query)
        results = [result.data for result in execute_query]
        if results:
            parsed_results = self.parse_existdb_results(results[0])
            if parsed_results:
                return parsed_results
            else:
                return None
        else:
            return None

    def find_consolidated(self, def_metadata):
        frbr_work = def_metadata['frbr_work']
        dataset = self.collection_names_map[def_metadata['dataset']]
        definendum_label = def_metadata['definendum_label']

        ARTICLE_REF_QUERY = r"""
            xquery version "3.1";
            declare namespace akn = "{namespace}";
            
            let $work:="{frbr_work}"
            let $aknShort := replace($work, '-\d{{2}}-\d{{2}}', '') 

            let $docs := collection('/db/EurLex-Consolidati')[replace(.//akn:FRBRWork/akn:FRBRuri/@value,"-\d{{2}}-\d{{2}}","")=$aknShort]

            let $results := 
                for $doc in $docs
                let $expdate := $doc//akn:FRBRExpression/akn:FRBRdate/@date/string()
                let $def := $doc//akn:definitionHead[@refersTo="{definendum_label}"]/@href/string()
                return
                    <result>
                        <date>{{$expdate}}</date>
                    {{for $definition in $doc//*[@defines=$def]
                        return <definition>{{$definition/string()}}</definition>}}
                    </result>

            return
                <results>{{$results}}</results>
        """

        namespace = self.namespaces[dataset]['akn']
        query = ARTICLE_REF_QUERY.format(
            namespace=namespace, frbr_work=frbr_work, definendum_label=definendum_label)

        execute_query = self._execute_query(query)
        results = [result.data for result in execute_query]
        return self.parse_existdb_results(results[0])
    
    def parse_existdb_results(self, xml_string):
        root = ET.fromstring(xml_string)
        results = root.findall('.//result')
        result_list = []
        for result in results:
            date = result.find('date')
            definition = result.find('definition')
            if definition.text is not None and date.text is not None and date.text != '' and definition.text != '':
                result_list.append({'date': datetime.strptime(date.text.split(' ')[0], '%Y-%m-%d'), 'definition': self.clean_exist_result(definition.text)})

        return result_list
    
    def clean_exist_result(self, text):
        return re.sub(r'\s+', ' ', text).strip()
    


def setup_existdb_handler():
    existdb_settings = settings.EXIST_CONFIG
    return ExistDBHandler(settings=existdb_settings)


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


def parse_date_filters(date_filters):
    from_date = date_filters[0]
    to_date = date_filters[1]
    
    if from_date == to_date:
        return datetime.strptime(from_date, '%Y-%m-%d').date()
    else:
        if not from_date:
            return datetime.strptime('0001-01-01', '%Y-%m-%d').date(), datetime.strptime(to_date, '%Y-%m-%d').date()
        elif not to_date:
            return datetime.strptime(from_date, '%Y-%m-%d').date(), datetime.today().date()
        else:
            return datetime.strptime(from_date, '%Y-%m-%d').date(), datetime.strptime(to_date, '%Y-%m-%d').date()


def doc_to_json(doc):
    json_doc = doc.to_json().get('kwargs')
    json_doc['definition_text'] = json_doc.pop('page_content')
    del json_doc['type']

    return json_doc


def docs_list_to_json_list(docs):
    json_list = [doc_to_json(doc) for doc in docs]

    return json_list


def get_frbr_uri_date(frbr_uri):
    date = re.search(r'(\d{4}-\d{2}-\d{2})', frbr_uri).group(1)

    return date


def json_to_aimessage(parsed_output):
    """Convert parsed output back to AIMessage."""
    content = f"{parsed_output}"
    return AIMessage(content=content)


def definition_obj_to_path(definition_obj):
    return os.path.join(definition_obj['dataset'], definition_obj['document_id'])


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


def get_uri_date(uri):
    at_split = uri.split("@")[-1]
    date = at_split.split("/")[0]
    return date


def setup_logging(log_level=logging.DEBUG):
    """
    Configures logging to output messages directly in Jupyter Notebook cells.

    Args:
        log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
    """
    # Remove any existing handlers attached to the root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure the logging system
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)  # Ensure logs are shown in notebook output
        ],
    )

    logging.info("Logging configured")


def filter_defs_list(original_list, relevant_ids):
    return [doc for doc in original_list if doc.metadata['id'] in relevant_ids]


def filter_documents_by_legislation(docs, legislation_filter):
    if not legislation_filter:
        return docs
    
    legislation_dataset_map = {
        'EU': ['EurLex'],
        'IT': ['Normattiva', 'PDL'],
    }
    
    return [doc for doc in docs if doc.metadata['dataset'] in legislation_dataset_map[legislation_filter]]


def get_frbr_expression_date(expression_uri):

    work_date = re.search(r'(\d{4}-\d{2}-\d{2})', expression_uri).group(1)

    return work_date


def filter_documents_by_date(docs, date_filters):
    if not date_filters:
        return docs
    
    date_filter = parse_date_filters(date_filters)

    filtered_defs = []
    for doc in docs:
        doc_date = datetime.strptime(get_frbr_expression_date(doc.metadata['frbr_expression']), '%Y-%m-%d').date()
        
        if isinstance(date_filter, date):
            if doc_date == date_filter:
                filtered_defs.append(doc)
        elif isinstance(date_filter, tuple):
            if date_filter[0] <= doc_date <= date_filter[1]:
                filtered_defs.append(doc)

    return filtered_defs


def retrieved_docs_list_to_dict(doc_list):
    return {
        doc.metadata['id']: {
            'definition_text': doc.page_content} 
            for doc in doc_list
            }


