import os
import streamlit as st
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv

from utils.txt_feature_extractor import extract_features_from_sample_battery_from_text

# ----------------------------------------------
# 1) Use langchain_neo4j for Neo4jGraph and GraphCypherQAChain
# ----------------------------------------------
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain

# ----------------------------------------------
# 2) Use NVIDIA endpoints for LLM and Embeddings
# ----------------------------------------------
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# ----------------------------------------------
# 3) Use Neo4j Vector Store from langchain_neo4j for Example Selection
# ----------------------------------------------
from langchain_neo4j import Neo4jVector

# ----------------------------------------------
# 4) Use LangChain's PromptTemplate and ExampleSelector
# ----------------------------------------------
from langchain.prompts import PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector

# --- Load environment variables ---
load_dotenv()

NVAPI_KEY = os.getenv("NVAPI_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NVAPI_KEY:
    st.error("⚠️ NVIDIA API key is missing! Please set it in your `.env` file.")
if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD:
    st.error("⚠️ Neo4j credentials are missing! Please set them in your `.env` file.")

# --- Initialize Neo4j Connection ---
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# --- Initialize NVIDIA LLM using ChatNVIDIA ---
llm = ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct",
    api_key=NVAPI_KEY,
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
)

# --- Initialize NVIDIA Embeddings using NVIDIAEmbeddings with model NV-Embed-QA ---
embedder = NVIDIAEmbeddings(
    model="NV-Embed-QA",
    api_key=NVAPI_KEY,
    truncate="NONE"
)

# --- Initialize Neo4j Vector Store for Semantic Similarity ---
vectorstore = Neo4jVector(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    embedding=embedder,
    pre_delete_collection=True  # Re-create the vector index with new embedding dimensions
)

# --- Example Prompts for Battery Data ---
examples = [
    {
        "question": "Which battery has the highest total cycles?",
        "query": "MATCH (b:Battery) RETURN b.battery_id, b.total_cycles ORDER BY b.total_cycles DESC LIMIT 1"
    },
    {
        "question": "Find batteries similar to one with slope_last_500_cycles = -0.000385",
        "query": "MATCH (b:Battery) WHERE abs(b.slope_last_500_cycles - (-0.000385)) < 0.0001 RETURN b.battery_id, b.slope_last_500_cycles"
    },
    {
        "question": "What is the charging policy of battery ID 'b1c19'?",
        "query": "MATCH (b:Battery {battery_id: 'b1c19'})-[:LINKED_TO]->(cp:chargingPolicy) RETURN cp.charging_policy"
    },
    {
        "question": "List all charging policies in the database.",
        "query": "MATCH (cp:chargingPolicy) RETURN cp.charging_policy"
    },
    {
        "question": "Which batteries have similar mean_grad_last_300_cycles?",
        "query": "MATCH (b:Battery) WHERE abs(b.mean_grad_last_300_cycles - (-0.000578)) < 0.0001 RETURN b.battery_id, b.mean_grad_last_300_cycles"
    },
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embedder,
    vectorstore,
    k=5,
    input_keys=["question"]
)

# --- Define the Prompt Template ---
CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher query to extract battery-related data from a Neo4j database.
Schema:
{schema}

Instructions:
- The schema is provided as key-value pairs.
- Identify the battery property mentioned in the question.
- Use the provided numeric values appropriately.
- Your response must be a comma-separated list of battery IDs only (no additional text).

Examples:
# Which battery has the highest total cycles?
MATCH (b:Battery) RETURN b.battery_id, b.total_cycles ORDER BY b.total_cycles DESC LIMIT 1

# Find batteries similar to one with slope_last_500_cycles = -0.000385
MATCH (b:Battery) WHERE abs(b.slope_last_500_cycles - (-0.000385)) < 0.0001 RETURN b.battery_id, b.slope_last_500_cycles

The query is:
{query}
"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "query"],
    template=CYPHER_GENERATION_TEMPLATE
)

# --- Create the GraphCypherQAChain instance ---
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    allow_dangerous_requests=True,
)

# --- Streamlit UI ---
st.title("🔋 Battery Data Query with AI (NVIDIA)")

# Variable to hold file-based schema (if file is uploaded)
file_schema = None

uploaded_file = st.file_uploader("Upload a battery data file (.txt)", type=["txt"])
if uploaded_file:
    file_content = uploaded_file.read().decode("utf-8")
    try:
        features = extract_features_from_sample_battery_from_text(file_content)
        file_schema = "\n".join(f"{key}: {value}" for key, value in features.items())
        st.success("✅ File uploaded and processed successfully!")
        st.write("📊 Extracted Features:")
        st.text(file_schema)
    except ValueError as e:
        st.error(f"⚠️ Error reading file: {e}")

user_query = st.text_input("🔍 Ask a question about battery features:")

if user_query:
    # Use file-based schema if available; otherwise, fallback to the graph's schema.
    if file_schema:
        schema_to_use = file_schema
    else:
        try:
            schema_to_use = str(graph.schema)
        except Exception:
            schema_to_use = "Battery nodes with properties like battery_id, total_cycles, slopes, etc."
    
    # (Optionally, retrieve relevant examples for debugging)
    relevant_examples = example_selector.select_examples({"question": user_query})
    response = chain.invoke({"schema": schema_to_use, "query": user_query})
    st.subheader("🔎 AI Response:")
    st.write(response)
