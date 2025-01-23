import re

from LegalDefAgent.src.utils import setup_existdb_handler
from LegalDefAgent.src.settings import settings


def clean_resolved_ref(binary_text: str) -> str:
    """
    Clean legal text by removing unnecessary whitespace, tabs, and quotes.
    
    Args:
        text (str): The input text to clean
        
    Returns:
        str: The cleaned text
    """

    text = binary_text.decode('utf-8')
    lines = text.split('\n')
    cleaned_text = ' '.join(lines)
    cleaned_text = ' '.join(cleaned_text.split())
    
    return cleaned_text.strip()

def resolve_reference(reference, dataset):
    REF_QUERY_EURLEX = r"""
    xquery version "3.1";
    declare namespace akn = "{namespace}";
    

    let $full_ref:="{reference}"
    let $split:=tokenize($full_ref,"~")

    let $aknShort:=replace($split[1],"(.*)(/ep)(.*)", "$1$3")
    let $aknShort:=replace($aknShort,"!main","")
    
    let $ref_el := $split[2]
    let $modified-text := replace($ref_el,"(\(([A-Za-z0-9]+)\))(.*)(\(([A-Za-z0-9]+)\))?", "__para_$2$3")
    let $eid := replace($modified-text,"(?:__point_)?\(([A-Za-z0-9]+)\)", "__list_1__point_$1")

    return collection("/db/{dataset}")[replace(.//akn:FRBRWork/akn:FRBRuri/@value,"-\d{{2}}-\d{{2}}","")=$aknShort]//*[matches(@eId,concat(".*(",$eid,")$"))][1]/string()
    """

    REF_QUERY_IT = r"""
    xquery version "3.1";
    declare namespace akn = "{namespace}";
    
    let $full_ref := "{reference}"
    let $split := tokenize($full_ref, "#")
    let $work := $split[1]

    let $ref := replace($split[2], "-com", "__para_")
    let $ref := replace($ref, "__item_", ".__point_")
    let $ref := replace($ref, "-let", ".__point_")
    let $ref := replace($ref, "bis", "-bis")

    let $split_ref := tokenize($ref, '__')
    let $rebuilt_ref := string-join(
        ($split_ref[1], "[^\d]*", subsequence($split_ref, 2) ! concat("__", .)),
        ""
    ) || "$"

    let $res := collection('/db/NormaAttiva/Documents')[.//akn:FRBRWork/akn:FRBRthis/@value = $work]//*[matches(@eId, $rebuilt_ref)]/string()

    return $res
    """

    exist_handler = setup_existdb_handler()
    ns = settings.DB_CONFIG.NAMESPACES
    coll_names = settings.EXIST_CONFIG.COLLECTION_NAMES_MAP
    namespace = ns[dataset]['akn']

    if dataset == 'EurLex':
        if '~' in reference and '__' in reference:
            query = REF_QUERY_EURLEX.format(
                namespace=namespace, reference=reference, dataset=coll_names[dataset])

            results = exist_handler.execute_query(query)
            if results:
                return clean_resolved_ref(results[0].data)
            else:
                return None
    else:
        if reference.startswith('/akn/') and any(c in reference.split('!')[1] for c in ['-', '__']):
            query = REF_QUERY_IT.format(
                namespace=namespace, reference=reference, dataset=dataset)

            results = exist_handler.execute_query(query)
            if results:
                return clean_resolved_ref(results[0].data)
            else:
               return None
    return None
