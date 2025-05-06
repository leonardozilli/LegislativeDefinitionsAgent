from datetime import datetime, date
from typing import List, Dict, Any, Union, Optional, Set, Tuple
import logging
import sys
import os
import re
from pathlib import Path

import tiktoken
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv, find_dotenv
from IPython.display import Image, display

load_dotenv(find_dotenv())


def setup_logging(log_level: int = logging.INFO) -> None:
    """Configure logging for Jupyter Notebook cells."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.info("Logging configured")


def get_token_count(text: str, model_name: str) -> int:
    """Count tokens for a given text using specified model's tokenizer."""
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))


def chatqa(query: str, model_name: str, temperature: float = 0) -> str:
    """Execute OpenAI inference through LangChain."""
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(template="{q}", input_variables=["q"])
    )
    return chain.apply([{'q': query}])[0]['text']


def merge_dicts(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries into one."""
    return {k: v for d in dicts for k, v in d.items()}


def parse_date_filters(date_filters: Tuple[str, str]) -> Union[date, Tuple[date, date]]:
    """Parse date filters into either a single date or a date range."""
    from_date, to_date = date_filters

    if from_date == to_date:
        return datetime.strptime(from_date, '%Y-%m-%d').date()

    if not from_date:
        start = datetime.strptime('0001-01-01', '%Y-%m-%d').date()
    else:
        start = datetime.strptime(from_date, '%Y-%m-%d').date()

    if not to_date:
        end = datetime.today().date()
    else:
        end = datetime.strptime(to_date, '%Y-%m-%d').date()

    return start, end


def parse_date(date_str: str) -> date:
    """Convert date string to date object."""
    return datetime.strptime(date_str, '%Y-%m-%d').date()


def doc_to_json(doc: Document) -> Dict[str, Any]:
    """Convert LangChain Document to JSON format."""
    json_doc = doc.to_json().get('kwargs')
    json_doc['definition_text'] = json_doc.pop('page_content')
    json_doc.pop('type', None)
    return json_doc


def docs_list_to_json_list(docs: List[Document]) -> List[Dict[str, Any]]:
    """Convert list of LangChain Documents to JSON list."""
    return [doc_to_json(doc) for doc in docs]


def extract_date_from_uri(uri: str) -> str:
    """Extract date from FRBR URI."""
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', uri)
    if not date_match:
        raise ValueError(f"No date found in URI: {uri}")
    return date_match.group(1)


def json_to_aimessage(parsed_output: Dict[str, Any]) -> AIMessage:
    """Convert parsed output to AIMessage format."""
    return AIMessage(content=str(parsed_output))


def definition_obj_to_path(definition_obj: Dict[str, str]) -> Path:
    """Convert definition object to filesystem path."""
    return Path(definition_obj['dataset']) / definition_obj['document_id']


def draw_graph(graph: Any) -> None:
    """Draw a Mermaid graph if dependencies are available."""
    try:
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    except Exception as e:
        logging.warning(f"Failed to draw graph: {e}")


def print_event(event: Dict[str, Any], printed_ids: Set[str], max_length: int = 1500) -> None:
    """Print event information with truncation if necessary."""
    if current_state := event.get("dialog_state"):
        print(f"Currently in: {current_state[-1]}")

    if message := event.get("messages"):
        if isinstance(message, list):
            message = message[-1]
        if message.id not in printed_ids:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = f"{msg_repr[:max_length]} ... (truncated)"
            print(msg_repr)
            printed_ids.add(message.id)


def filter_definitions(docs: List[Document],
                       relevant_ids: Optional[List[str]] = None,
                       legislation: Optional[str] = None) -> List[Document]:
    """Filter definitions based on IDs and legislation."""
    filtered_docs = docs

    if relevant_ids:
        filtered_docs = [doc for doc in filtered_docs
                         if doc.metadata['id'] in relevant_ids]

    if legislation:
        legislation_datasets = {
            'EU': ['EurLex'],
            'IT': ['Normattiva', 'PDL'],
        }
        filtered_docs = [doc for doc in filtered_docs
                         if doc.metadata['dataset'] in legislation_datasets[legislation]]

    return filtered_docs


def filter_documents_by_date(docs: List[Document],
                             date_filters: Optional[Tuple[str, str]]) -> List[Document]:
    """Filter documents based on date criteria."""
    if not date_filters:
        return docs

    date_filter = parse_date_filters(date_filters)
    filtered_docs = []

    for doc in docs:
        doc_date = parse_date(extract_date_from_uri(
            doc.metadata['frbr_expression']))

        if isinstance(date_filter, date):
            if doc_date == date_filter:
                filtered_docs.append(doc)
        elif isinstance(date_filter, tuple):
            if date_filter[0] <= doc_date <= date_filter[1]:
                filtered_docs.append(doc)

    return filtered_docs


def format_definitions_dict(data: List[Dict[str, Any]],
                       include_keywords: bool = True) -> str:
    """Format definition data into readable string."""
    formatted_parts = []

    for entry in data:
        metadata = entry['metadata']
        parts = [
            f"ID: {metadata['id']}",
            f"Dataset: {metadata['dataset']}",
            "\nTimeline:"
        ]

        for i, timeline_entry in enumerate(entry['timeline'], 1):
            parts.extend([
                f"{i}. Date: {timeline_entry['date']}",
                f"   Definition: {timeline_entry['definition']}\n"
            ])

        if include_keywords and 'keywords' in entry:
            parts.extend([
                "Keywords:",
                *[f"  - {kw}" for kw in entry['keywords']]
            ])

        formatted_parts.append("\n".join(parts))
        formatted_parts.append("-" * 50)

    return "\n\n".join(formatted_parts)


def format_definitions_dict_xml(data, include_keywords: bool = True) -> str:
    """Format definition data into XML-like string."""
    formatted_parts = []
    
    for entry in data:
        parts = [
            "<definition>",
            f"  <definition_id>{entry['metadata']['id']}</definition_id>",
            f"  <dataset>{entry['metadata']['dataset']}</dataset>",
            "  <timeline>",
        ]
        
        for i, timeline_entry in enumerate(entry['timeline'], 1):
            parts.extend([
                "    <entry>",
                f"      <entry_id>{i}</entry_id>",
                f"      <date>{timeline_entry['date']}</date>",
                f"      <definition_text>{timeline_entry['definition']}</definition_text>",
                "    </entry>"
            ])
        
        parts.append("  </timeline>")
        
        if include_keywords and 'keywords' in entry:
            parts.append(f"  <keywords>{', '.join(entry['keywords'])}</keywords>")
        
        parts.append("</definition>")
        formatted_parts.append("\n".join(parts))
    
    return "\n\n".join(formatted_parts)


def format_answer_definition(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format answer definition with specific timeline entry."""
    return [{
        'dataset': entry['metadata']['dataset'],
        'document_id': entry['metadata']['document_id'].split('.')[0],
        'definition': [tl_entry for tl_entry in entry['timeline']]
    } for entry in data]


def camelcase_to_spaces(text: str) -> str:
    """Convert camelCase to space-separated lowercase text."""
    text = text.replace('#', '')
    text = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', text)
    return text.lower()


def parse_date_string(s):
    start_date, end_date = s.split(' - ')
    start_date = None if start_date == 'None' else start_date
    end_date = None if end_date == 'None' else end_date

    return [start_date, end_date]
