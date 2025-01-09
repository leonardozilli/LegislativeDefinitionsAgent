from ..settings import settings
from typing import List, Optional, Any
from pyexistdb import db

from datetime import datetime


existdb_settings = settings.EXIST_CONFIG

class ExistDBHandler:
    """Handler for executing XQueries against an eXist-db instance."""
    
    def __init__(self, server_url: str, username: str, password: str):
        """
        Initialize connection to eXist-db.
        
        Args:
            server_url: Full URL to eXist-db server
            username: eXist-db username
            password: eXist-db password
        """
        self.db = db.ExistDB(server_url, username, password)
    
    def execute_query(self, query: str) -> List[str]:
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


EXISTDB_SERVER_URL = f"http://{existdb_settings.XDB_HOST}:{existdb_settings.XDB_PORT}/exist/"

handler = ExistDBHandler(
    server_url=EXISTDB_SERVER_URL,
    username=existdb_settings.XDB_USER,
    password=existdb_settings.XDB_PASSWORD
)

GET_DOCUMENT_DATE_QUERY = r"""
    xquery version "3.1";
    declare namespace akn = "{namespace}";
    let $exp := "{expression}"
    let $exps_res := collection('/db')[.//akn:FRBRExpression/akn:FRBRthis/@value=$exp]//akn:FRBRExpression
    return $exps_res[1]/akn:FRBRdate/@date/string()
"""

def retrieve_doc_date(doc):
    dataset = doc.metadata['dataset']
    namespace = settings.DB_CONFIG.NAMESPACES[dataset]['akn']
    expression = doc.metadata['frbr_expression']
    
    query = GET_DOCUMENT_DATE_QUERY.format(
        namespace=namespace,
        expression=expression
    )
    
    results = handler.execute_query(query)
    if results:
        return datetime.strptime(str(results[0]), '%Y-%m-%d').date()
    return None