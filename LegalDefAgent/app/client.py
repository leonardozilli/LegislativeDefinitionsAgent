import streamlit as st
from langserve import RemoteRunnable
from pprint import pprint
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage, AnyMessage

st.title('Welcome to Speckle Server')
input_text = st.text_input('ask speckle related question here')

inputs = {
    "input": input_text,
    "messages": [
        SystemMessage(content=''),
        HumanMessage(content=input_text)
    ]
}

if input_text:
    with st.spinner("Processing..."):
        try:
            app = RemoteRunnable("http://localhost:8000/defagent/")
            for output in app.stream(inputs):
                pprint(output)
                    # Optional: print full state at each node
                    # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
                pprint("\n---\n")
            st.write(output)
        
        except Exception as e:
            st.error(f"Error: {e}")