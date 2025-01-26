from typing import List
from pyexistdb import db
from datetime import datetime
import xml.etree.ElementTree as ET
import re

from ..settings import settings


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
                result_list.append({'date': datetime.strptime(date.text.split(
                    ' ')[0], '%Y-%m-%d'), 'definition': self.clean_exist_result(definition.text)})

        return result_list
    
    def resolve_reference(self, reference, dataset):
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

        dataset = self.collection_names_map[dataset]
        namespace = self.namespaces[dataset]['akn']
    
        if dataset == 'EurLex':
            if '~' in reference and '__' in reference:
                query = REF_QUERY_EURLEX.format(
                    namespace=namespace, reference=reference, dataset=dataset)

                results = self._execute_query(query)
                if results:
                    return self.clean_resolved_ref(results[0].data)
                else:
                    return None
        else:
            if reference.startswith('/akn/') and any(c in reference.split('!')[1] for c in ['-', '__']):
                query = REF_QUERY_IT.format(
                    namespace=namespace, reference=reference, dataset=dataset)

                results = self._execute_query(query)
                if results:
                    return self.clean_resolved_ref(results[0].data)
                else:
                    return None
        return None

    def clean_resolved_ref(self, binary_text: str) -> str:
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

    def clean_exist_result(self, text):
        return re.sub(r'\s+', ' ', text).strip()


existdb_handler = ExistDBHandler(settings=settings.EXIST_CONFIG)
