from langchain_core.messages import AIMessage
import streamlit as st

import LegalDefAgent.src.config as config
import LegalDefAgent.src.models as models
from LegalDefAgent.src.agent import LegalDefAgent



async def invoke_graph(agent, st_messages, st_placeholder, config=None, debug=False):
    """
    Asynchronously processes a stream of events from the graph_runnable and updates the Streamlit interface.

    Args:
        st_messages (list): List of messages to be sent to the graph_runnable.
        st_placeholder (st.beta_container): Streamlit placeholder used to display updates and statuses.

    Returns:
        AIMessage: An AIMessage object containing the final aggregated text content from the events.
    """
    # Set up placeholders for displaying updates in the Streamlit app
    container = st_placeholder  # This container will hold the dynamic Streamlit UI components
    thoughts_placeholder = container.container()  # Container for displaying status messages
    token_placeholder = container.empty()  # Placeholder for displaying progressive token updates
    final_text = ""  # Will store the accumulated text from the model's response
    other_text = ""

    graph_runnable = agent.graph_runnable
    output_nodes = {'answer', 'model', 'ask_to_generate'}
    graph_nodes = graph_runnable.nodes.keys() - output_nodes - {'__start__', 'task_manager', 'query_vectorstore'}

    print_output = False
    column_width = 20

    # Stream events from the graph_runnable asynchronously
    async for event in graph_runnable.astream_events({"messages": st_messages}, version="v2", config=config, subgraphs=False):
        kind = event["event"]  # Determine the type of event received
        name = event["name"]  # Determine the name of the event received
        if debug:
            tags = ", ".join(event["tags"])  # Convert list to comma-separated string
            data = str(event.get("data", {}))
            if len(data) > 80:
                data = data[:50] + "..."
            print(f'{event["event"]:<{column_width}} | {name:<{column_width}} | {tags:<{30}} | {data}')

        if kind == "on_chain_start" and name in graph_nodes:
            # The event signals the start of a chain execution
            with thoughts_placeholder:
                status_placeholder = st.empty()  # Placeholder to show the tool's status
                with status_placeholder.status(f"Calling node {name}...", expanded=True) as s:
                    st.write("Executed", event['name'])  # Show which tool is being called
                    st.write("Node input: ")
                    st.code(event['data'].get('input')['messages'][-1].content, wrap_lines=True)  # Display the input data sent to the tool
                    st.write("Node output: ")
                    output_placeholder = st.empty()  # Placeholder for tool output that will be updated later below
                    s.update(label=f"Executed node {name}...", expanded=False)  # Update the status once done
        elif kind == "on_chain_end" and name in graph_nodes:
            with thoughts_placeholder:
                # We assume that `on_tool_end` comes after `on_tool_start`, meaning output_placeholder exists
                if 'output_placeholder' in locals():
                    output_placeholder.code(event['data'].get('output')['messages'][-1].content, wrap_lines=True)  # Display the tool's output


        #vectorstore hack
        if kind == "on_chain_start" and name == 'query_vectorstore':
            # The event signals the start of a chain execution
            with thoughts_placeholder:
                status_placeholder = st.empty()  # Placeholder to show the tool's status
                with status_placeholder.status(f"Calling node {name}...", expanded=True) as s:
                    st.write("Executed", event['name'])  # Show which tool is being called
                    st.write("Node input: ")
                    st.code(event['data'].get('input')['messages'][-1].content, wrap_lines=True)  # Display the input data sent to the tool
                    st.write("Node output: ")
                    output_placeholder = st.empty()  # Placeholder for tool output that will be updated later below
                    s.update(label=f"Executed node {name}...", expanded=False)  # Update the status once done
 
        elif kind == "on_chain_end" and name == 'query_vectorstore':
            with thoughts_placeholder:
                # We assume that `on_tool_end` comes after `on_tool_start`, meaning output_placeholder exists
                if 'output_placeholder' in locals():
                    output_placeholder.code(event['data'].get('output')['retrieved_definitions'], wrap_lines=True)  # Display the tool's output

        ###
        
        elif kind == "on_chain_start" and name in output_nodes:
            print_output = True

        elif print_output and kind == "on_chat_model_stream":
            # The event corresponding to a stream of new content (tokens or chunks of text)
            addition = event["data"]["chunk"].content  # Extract the new content chunk
            final_text += addition  # Append the new content to the accumulated text
            if addition:
                token_placeholder.write(final_text)  # Update the st placeholder with the progressive response

        elif print_output and kind == "on_chain_stream" and name == 'ask_to_generate':
            addition = event["data"]["chunk"]['messages'][0].content  # Extract the new content chunk
            other_text += addition  # Append the new content to the accumulated text
            if addition:
                token_placeholder.write(other_text)  # Update the st placeholder with the progressive response

        elif kind == "on_tool_start":
            # The event signals that a tool is about to be called
            with thoughts_placeholder:
                status_placeholder = st.empty()  # Placeholder to show the tool's status
                with status_placeholder.status("Calling Tool...", expanded=True) as s:
                    st.write("Called ", event['name'])  # Show which tool is being called
                    st.write("Tool input: ")
                    st.code(event['data'].get('input'))  # Display the input data sent to the tool
                    st.write("Tool output: ")
                    output_placeholder = st.empty()  # Placeholder for tool output that will be updated later below
                    s.update(label="Completed Calling Tool!", expanded=False)  # Update the status once done

        elif kind == "on_tool_end":
            # The event signals the completion of a tool's execution
            with thoughts_placeholder:
                # We assume that `on_tool_end` comes after `on_tool_start`, meaning output_placeholder exists
                if 'output_placeholder' in locals():
                    output_placeholder.code(event['data'].get('output').content)  # Display the tool's output    # Return the final aggregated message after all events have been processed
    return final_text
