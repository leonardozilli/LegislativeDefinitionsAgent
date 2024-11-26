from string import Template
from langchain_core.prompts import PromptTemplate


SYSTEM_PROMPT = Template("""
You are a legal assistant with expertise in legal definitions. Your task is to help find the right definition for a given term.
""")


refine_answers_prompt = Template("""
${raw_answers}

Given the above dictionary, that follows the structure of {'id': {'column_name(column_data_type)': 'column_value'}}, homogenize the column_values,
respecting the data type of the column provided.

Some rules for the homogenization include but are not limited to: 
Values must not be lists. Dates must be in the format of dd/mm/YYYY. Boolean values must be either True or False, case sensitive.

Reply only with the homogenized dictionary.

""")


generate_definition_prompt = Template("""
${raw_answers}

Given the above dictionary, that follows the structure of {'id': {'column_name(column_data_type)': 'column_value'}}, homogenize the column_values,
respecting the data type of the column provided.

Some rules for the homogenization include but are not limited to: 
Values must not be lists. Dates must be in the format of dd/mm/YYYY. Boolean values must be either True or False, case sensitive.

Reply only with the homogenized dictionary.

""")

rag_prompt = PromptTemplate(
    template="""
        You are an assistant for question-answering tasks. \n
        Use the following pieces of retrieved context to answer the question. \n
        If you don't know the answer, just say that you don't know. \n
        Use three sentences maximum and keep the answer concise.
        Question: {question} \n
        Context: {context} \n
        Answer:
        """,
    input_variables=["context", "question"]
)
