import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List

from pyexistdb import db

from legaldefagent.settings import Settings, settings


class ExistDBHandler:
    """Handler for executing XQueries against an eXist-db instance."""

    def __init__(self, settings: Settings):
        self.xdb_config = settings.existdb
        self.db = db.ExistDB(
            server_url=f"http://{self.xdb_config.host}:{self.xdb_config.port}/exist/",
            username=self.xdb_config.user,
            password=self.xdb_config.password,
        )
        self.collections = settings.collections
        self.consolidated_collection = settings.consolidated_collection

    def _execute_query(self, query: str) -> List:
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

    def get_work_eurovocs(self, def_metadata: dict) -> list[dict]:
        frbr_work = def_metadata["frbr_work"]
        dataset = def_metadata["dataset"]
        definiendum_label = def_metadata["definiendum_label"]
        QUERY = r"""
            xquery version "3.1";
            declare namespace akn = "{namespace}";

            let $work:="{frbr_work}"

            let $doc := collection('/db/{dataset}')[.//akn:FRBRWork/akn:FRBRthis/@value=$work]


            return $doc/*//akn:classification/akn:keyword/@showAs/string()
        """

        namespace = self.collections[dataset].namespace
        query = QUERY.format(
            namespace=namespace,
            dataset=dataset,
            frbr_work=frbr_work,
            definiendum_label=definiendum_label,
        )

        execute_query = self._execute_query(query)
        results = [result.data.decode() for result in execute_query]

        return results

    def extract_definition_from_exist(self, def_metadata: dict) -> list[dict] | None:
        frbr_work = def_metadata["frbr_work"]
        dataset = def_metadata["dataset"]
        definiendum_label = def_metadata["definiendum_label"]
        if def_metadata["dataset"] == "EurLex":
            ARTICLE_REF_QUERY = r"""
                xquery version "3.1";
                declare namespace akn = "{namespace}";

                let $work:="{frbr_work}"

                let $docs := collection('/db/{dataset}')[.//akn:FRBRWork/akn:FRBRthis/@value=$work]

                let $results :=
                    for $doc in $docs
                    let $expdate := $doc//akn:FRBRExpression/akn:FRBRdate/@date/string()
                    let $def := $doc//akn:definitionHead[@refersTo="{definiendum_label}"]/@href/string()
                    return
                        <result>
                            <date>{{$expdate}}</date>
                        {{for $definition in $doc//*[@defines=$def]
                            return <definition>{{$definition/string()}}</definition>}}
                        </result>

                return
                    <results>{{$results}}</results>
            """
        elif def_metadata["dataset"] == "Normattiva":
            ARTICLE_REF_QUERY = r"""
                xquery version "3.1";
                declare namespace akn = "{namespace}";

                let $work:="{frbr_work}"

                let $docs := collection('/db/{dataset}')[.//akn:FRBRWork/akn:FRBRthis/@value=$work]

                let $results :=
                    for $doc in $docs
                    let $expdate := $doc//akn:FRBRExpression/akn:FRBRdate/@date/string()
                    let $def := $doc//akn:definitionHead[@refersTo="{definiendum_label}"]/@href/string()
                    return
                        <result>
                            <date>{{$expdate}}</date>
                        {{for $definition in $doc//*[@defines=$def]
                            return <definition>{{$definition/string()}}</definition>}}
                        </result>

                return
                    <results>{{$results}}</results>
            """
        else:
            ARTICLE_REF_QUERY = r"""
                xquery version "3.1";
                declare namespace akn = "{namespace}";

                let $work:="{frbr_work}"

                let $docs := collection('/db/{dataset}')[.//akn:FRBRWork/akn:FRBRthis/@value=$work]

                let $results :=
                    for $doc in $docs
                    let $expdate := replace($doc//akn:FRBRExpression/akn:FRBRuri/@value/string(), ".*/(\d{{4}}-\d{{2}}-\d{{2}}).*", "$1")
                    let $def := $doc//akn:definitionHead[@refersTo="{definiendum_label}"]/@href/string()
                    return
                        <result>
                            <date>{{$expdate}}</date>
                        {{for $definition in $doc//*[@defines=$def]
                            return <definition>{{$definition/string()}}</definition>}}
                        </result>

                return
                    <results>{{$results}}</results>
            """

        namespace = self.collections[dataset].namespace
        query = ARTICLE_REF_QUERY.format(
            namespace=namespace,
            dataset=dataset,
            frbr_work=frbr_work,
            definiendum_label=definiendum_label,
        )

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

    def get_all_definitions(self, dataset: str) -> str:
        QUERY = r"""
                xquery version "3.1";
                declare namespace akn = "{namespace}";
                declare namespace output = "http://www.w3.org/2010/xslt-xquery-serialization";
                declare option output:method "json";
                declare option output:media-type "application/json";

                let $dataset := "{dataset}"

                return <results>
                {{
                    for $doc in collection(concat("/db/", $dataset))
                    let $frbr_work := $doc//akn:FRBRWork/akn:FRBRthis/@value/string()
                    let $frbr_expression := $doc//akn:FRBRExpression/akn:FRBRthis/@value/string()
                    let $keywords := $doc//akn:keyword/@showAs/string()

                    for $definition in $doc//akn:definition
                    let $definiendum_id := $definition/akn:definitionHead/@href/string() => substring-after('#')
                    let $definiendum_raw := ($doc//akn:def[@eId = $definiendum_id])[1]/string()
                    let $definiens_parts := for $body_ref in $definition/akn:definitionBody
                                        let $body_id := $body_ref/@href/string() => substring-after('#')
                                        return $doc//akn:defBody[@eId = $body_id]

                    let $references := $definiens_parts//akn:ref/@href/string()

                    let $full_def_raw := ($doc//*[@defines = concat('#', $definiendum_id)])[1]/string()

                    let $cleaned_definiendum := string($definiendum_raw) => normalize-space() => replace('^["''«]|["''»]$', '')
                    let $cleaned_definiens   := $definiens_parts/string() => string-join(" ") => normalize-space() => replace('^[:,\s]+', '')
                    let $cleaned_full_def    := $full_def_raw => normalize-space() => replace('""', '')

                    return
                        <definition>
                            <def_n>{{$definition/akn:definitionHead/@href/string()}}</def_n>
                            <label>{{$definition/akn:definitionHead/@refersTo/string()}}</label>
                            <definiendum>{{$cleaned_definiendum}}</definiendum>
                            <definiens>{{$cleaned_definiens}}</definiens>
                            <full_definition>{{$cleaned_full_def}}</full_definition>
                            <references>{{string-join($references, ', ')}}</references>
                            <provenance>{{$dataset}}</provenance>
                            <document>{{document-uri($doc) => tokenize('/') => reverse() => head()}}</document>
                            <frbr_work>{{$frbr_work[1]}}</frbr_work>
                            <frbr_expression>{{$frbr_expression[1]}}</frbr_expression>
                            <keywords>{{string-join($keywords, ", ")}}</keywords>
                        </definition>
                }}
                </results>
            """

        namespace = self.collections[dataset].namespace
        query = QUERY.format(
            namespace=namespace,
            dataset=dataset,
        )
        results = self._execute_query(query)
        xml_strings = [result.data.decode("utf-8") for result in results]

        return xml_strings[0]  # [self.parse_existdb_results(xml) for xml in xml_strings]

    def find_consolidated(self, def_metadata):
        frbr_work = def_metadata["frbr_work"]
        dataset = def_metadata["dataset"]
        definiendum_label = def_metadata["definiendum_label"]

        ARTICLE_REF_QUERY = r"""
            xquery version "3.1";
            declare namespace akn = "{namespace}";

            let $work:="{frbr_work}"
            let $aknShort := replace($work, '-\d{{2}}-\d{{2}}', '')

            let $docs := collection("/db/{consolidated_collection}")[replace(.//akn:FRBRWork/akn:FRBRuri/@value,"-\d{{2}}-\d{{2}}","")=$aknShort]

            let $results :=
                for $doc in $docs
                let $expdate := $doc//akn:FRBRExpression/akn:FRBRdate/@date/string()
                let $def := $doc//akn:definitionHead[@refersTo="{definiendum_label}"]/@href/string()
                return
                    <result>
                        <date>{{$expdate}}</date>
                    {{for $definition in $doc//*[@defines=$def]
                        return <definition>{{$definition/string()}}</definition>}}
                    </result>

            return
                <results>{{$results}}</results>
        """

        namespace = self.collections[dataset].namespace
        query = ARTICLE_REF_QUERY.format(
            namespace=namespace,
            frbr_work=frbr_work,
            definiendum_label=definiendum_label,
            consolidated_collection=self.consolidated_collection,
        )

        execute_query = self._execute_query(query)
        results = [result.data for result in execute_query]
        return self.parse_existdb_results(results[0])

    def parse_existdb_results(self, xml_string) -> list[dict]:
        root = ET.fromstring(xml_string)
        results = root.findall(".//result")
        result_list = []
        for result in results:
            date = result.find("date")
            definition = result.find("definition")
            if definition is not None:
                if definition.text is not None and date.text is not None and date.text != "" and definition.text != "":
                    result_list.append(
                        {
                            "date": datetime.strptime(date.text.split(" ")[0], "%Y-%m-%d"),
                            "definition": self.clean_exist_result(definition.text),
                        }
                    )

        return result_list

    def resolve_reference(self, reference, dataset):
        REF_QUERY_EURLEX = r"""
        xquery version "3.1";
        declare namespace akn = "{namespace}";

        let $full_ref := "{reference}"
        let $split := tokenize($full_ref,"~")

        let $aknShort := replace($split[1],"(.*)(/ep)(.*)", "$1$3")
        let $aknShort := replace($aknShort,"!main","")

        let $ref_el := $split[2]
        let $modified-text := replace($ref_el,"(\(([A-Za-z0-9]+)\))(.*)(\(([A-Za-z0-9]+)\))?", "__para_$2$3")
        let $eid := replace($modified-text,"(?:__point_)?\(([A-Za-z0-9]+)\)", "__list_1__point_$1")

        let $res := collection("/db/{dataset}")[replace(.//akn:FRBRWork/akn:FRBRuri/@value,"-\d{{2}}-\d{{2}}","")=$aknShort]//*[matches(@eId,concat(".*(",$eid,")$"))][1]/string()

        return $res
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

        namespace = self.collections[dataset].namespace

        if dataset == "EurLex":
            if "~" in reference and "__" in reference:
                query = REF_QUERY_EURLEX.format(namespace=namespace, reference=reference, dataset=dataset)

                results = self._execute_query(query)
                if results:
                    return self.clean_resolved_ref(results[0].data)
                else:
                    return None
        else:
            if reference.startswith("/akn/") and any(c in reference.split("!")[1] for c in ["-", "__"]):
                query = REF_QUERY_IT.format(namespace=namespace, reference=reference, dataset=dataset)

                results = self._execute_query(query)
                if results:
                    return self.clean_resolved_ref(results[0].data)
                else:
                    return None
        return None

    def clean_resolved_ref(self, binary_text: bytes) -> str:
        text = binary_text.decode("utf-8")
        lines = text.split("\n")
        cleaned_text = " ".join(lines)
        cleaned_text = " ".join(cleaned_text.split())

        return cleaned_text.strip()

    def clean_exist_result(self, text):
        return re.sub(r"\s+", " ", text).strip()


existdb_handler = ExistDBHandler(settings=settings)
