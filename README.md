
# SciAI
This is a Lang-graph based project that using LLMs with data gathered from Arxiv and Semantic Scholar to  generate scientifically accurate reports.


# Features
- Scientific Query Generation:
        Generates specific search queries based on user input using structured output from the LLM.

- Source Aggregation:
        Aggregates scientific papers and abstracts from Arxiv and Semantic Scholar.
- Data Synthesis:
        Synthesizes collected data into meaningful insights using the LLM.
- Streamlit Integration:
        Provides a user-friendly interface for input and report generation.
- Downloadable Reports:
        Allows users to download synthesized reports in text format.

# Graph Structure

              START
                |
                v
            query_construct
                |
                v
            source_aggregator
                |
                v
            data_synthesis
                |
                v
               END

# Techstack
- Python
- Lang-graph
- Langchain document loaders
- Streamlit
- Arxiv API
- Semantic Scholar API


