from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import Annotated
from langchain_community.document_loaders import ArxivLoader
from semanticscholar import SemanticScholar
from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
import streamlit as st
import tempfile
import os


load_dotenv()

llm = init_chat_model(
    "groq:llama3-8b-8192",
    api_key = st.secrets["GROQ_API_KEY"]
)



Arxiv = []
ss = []
s = SemanticScholar()
sch = SemanticScholarAPIWrapper(
    semanticscholar_search=s,
    top_k_results=3,
    load_max_docs=3
)


#setting up state graph
class State(TypedDict):
    messages: Annotated[list, add_messages]
    search_queries_arxiv: list[str]
    search_queries_semanticscholar: list[str]

class Search_query(BaseModel):
    queries_arxiv: list[str] = Field(..., description="list of queries for serching arxiv")
    queries_semanticscholar: list[str] = Field(..., description="list of queries for serching semanticscholar")

graph_builder = StateGraph(State)

def query_construct(state: State):
    structured_llm = llm.with_structured_output(Search_query)
    last_message = state["messages"][-1]
    user_message = last_message.content if hasattr(last_message, "content") else last_message["content"]

    prompt = f"Based on this user request: '{user_message}', generate 2-3 specific search queries for finding relevant scientific papers."

    search_query_obj = structured_llm.invoke(prompt)

    # Safely extract queries whether search_query_obj is a BaseModel or dict
    if isinstance(search_query_obj, dict):
        queries_arxiv = search_query_obj.get("queries_arxiv", [])
        queries_semanticscholar = search_query_obj.get("queries_semanticscholar", [])
    else:
        queries_arxiv = getattr(search_query_obj, "queries_arxiv", [])
        queries_semanticscholar = getattr(search_query_obj, "queries_semanticscholar", [])
    return {"search_queries_arxiv": queries_arxiv, "search_queries_semanticscholar": queries_semanticscholar}


def source_aggregator(state:State):
    queries = state.get("search_queries_arxiv", [])
    queries_SS = state.get("search_queries_semanticscholar", [])
    for q in queries:
        try:
            print(f"Searching for: {q}")
            st.write(f"Searching for: {q}")
            loader = ArxivLoader(
                query=q,
                load_max_docs=1,
            )
            docs = loader.get_summaries_as_docs()
            Arxiv.append(docs)
            info_message = f"Information gathered: {Arxiv}"
            print(info_message)
        except Exception as e:
            print(f"Error occurred: {e}")
            st.write(f"Error occurred: {e}")
            Arxiv.append(f"Error occurred: {e}")

    for qs in queries_SS:
        print(f"Searching for: {qs}")
        st.write(f"Searching for: {qs}")
        r = sch.run(qs)
        if isinstance(r, dict):
            ss.append(r.get('abstract', ''))
        else:
            ss.append(None)
        print(r)
        info_message_S = f"Information gathered from other sources: {ss}"
        print(info_message_S)
    
    
    combined_info = f"Arxiv: {Arxiv}\nSemanticScholar: {ss}"
    print("-----------combined-info-----------",combined_info)
    return {"messages": [{"role": "system", "content": combined_info}]}


def data_synthesis(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


st.title("SciAI")
usr_inp = st.text_input("Enter your query:")


if st.button("Analyze") and usr_inp != None:
    with st.spinner("Preparing your report..."):
        #building graph
        graph_builder.add_node("query_construct", query_construct)
        graph_builder.add_node("source_aggregator", source_aggregator)
        graph_builder.add_node("data_synthesis", data_synthesis)

        graph_builder.add_edge(START, "query_construct")
        graph_builder.add_edge("query_construct", "source_aggregator")
        graph_builder.add_edge("source_aggregator", "data_synthesis")
        graph_builder.add_edge("data_synthesis", END)

        #compile
        graph = graph_builder.compile()


        state = graph.invoke({"messages": [{"role": "user", "content": usr_inp}], "search_queries": []})
        print(state['messages'][-1].content)
        parsed_resp = state['messages'][-1].content
        print(parsed_resp)
        st.write(parsed_resp)

        report_file = tempfile.TemporaryFile()

    # Encode the string to bytes before writing
    report_bytes = parsed_resp.encode('utf-8')

    st.download_button(
    label="Download Report",
    data=report_bytes,
    file_name="report.txt",
    mime="text/csv",
    icon=":material/download:",)

