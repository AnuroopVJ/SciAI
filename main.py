
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
import os

from io import BytesIO
import subprocess
import sys
# Load environment variables
load_dotenv()

# Auth key
api_key = st.secrets["GROQ_API_KEY"]

# Initialize LLM
llm = init_chat_model(
    "groq:llama-3.1-8b-instant2",
    api_key=api_key
)

if api_key:
    print("Auth_key_found")
    st.info("Auth_key_found")

Arxiv = []
ss = []
s = SemanticScholar()
sch = SemanticScholarAPIWrapper(
    semanticscholar_search=s,
    top_k_results=3,
    load_max_docs=3
)

# --- PDF Generator ---
def parse_headings_and_body(text):
    paragraphs = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("**") and "**" in line:
            heading = line[3:line.index("]**")]
            paragraphs.append(("heading", heading))
        elif line:
            paragraphs.append(("body", line))
    return paragraphs


# --- LangGraph State Definitions ---
class State(TypedDict):
    messages: Annotated[list, add_messages]
    search_queries_arxiv: list[str]
    search_queries_semanticscholar: list[str]

class Search_query(BaseModel):
    queries_arxiv: list[str] = Field(..., description="list of queries for serching arxiv")
    queries_semanticscholar: list[str] = Field(..., description="list of queries for serching semanticscholar")

# Build LangGraph
graph_builder = StateGraph(State)

def query_construct(state: State):
    structured_llm = llm.with_structured_output(Search_query)
    last_message = state["messages"][-1]
    user_message = last_message.content if hasattr(last_message, "content") else last_message["content"]

    prompt = f"Based on this user request: '{user_message}', generate 2-3 specific search queries for finding relevant scientific papers."

    search_query_obj = structured_llm.invoke(prompt)

    if isinstance(search_query_obj, dict):
        queries_arxiv = search_query_obj.get("queries_arxiv", [])
        queries_semanticscholar = search_query_obj.get("queries_semanticscholar", [])
    else:
        queries_arxiv = getattr(search_query_obj, "queries_arxiv", [])
        queries_semanticscholar = getattr(search_query_obj, "queries_semanticscholar", [])

    return {"search_queries_arxiv": queries_arxiv, "search_queries_semanticscholar": queries_semanticscholar}

def source_aggregator(state: State):
    queries = state.get("search_queries_arxiv", [])
    queries_SS = state.get("search_queries_semanticscholar", [])

    for q in queries:
        try:
            st.write(f"Searching Arxiv for: {q}")
            loader = ArxivLoader(query=q, load_max_docs=1)
            docs = loader.get_summaries_as_docs()
            Arxiv.append(docs)
        except Exception as e:
            st.write(f"Error: {e}")
            Arxiv.append(f"Error: {e}")

    for qs in queries_SS:
        st.write(f"Searching SemanticScholar for: {qs}")
        r = sch.run(qs)
        ss.append(r.get('abstract', '') if isinstance(r, dict) else None)

    combined_info = f"Arxiv: {Arxiv}\nSemanticScholar: {ss}"
    return {"messages": [{"role": "system", "content": combined_info}]}

def data_synthesis(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# --- Streamlit UI ---
st.title("SciAI")
usr_inp = st.text_input("Enter your query:")

if st.button("Analyze") and usr_inp:
    with st.spinner("Preparing your report..."):
        # Build graph
        graph_builder.add_node("query_construct", query_construct)
        graph_builder.add_node("source_aggregator", source_aggregator)
        graph_builder.add_node("data_synthesis", data_synthesis)

        graph_builder.add_edge(START, "query_construct")
        graph_builder.add_edge("query_construct", "source_aggregator")
        graph_builder.add_edge("source_aggregator", "data_synthesis")
        graph_builder.add_edge("data_synthesis", END)

        graph = graph_builder.compile()
        state = graph.invoke({"messages": [{"role": "user", "content": usr_inp}], "search_queries": []})

        parsed_resp = state['messages'][-1].content
        st.write(parsed_resp)
        text_file = BytesIO(parsed_resp.encode('utf-8'))

        # Download button
        st.download_button(
            label="Download Report (txt)",
            data=text_file,
            file_name="Report.txt",
            mime="text/plain",
            icon="📄",
        )
