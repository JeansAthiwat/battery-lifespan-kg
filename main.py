# ───────────────────────────────────────────────────────
# 🌐 System & Environment
# ───────────────────────────────────────────────────────
import os
import re
from dotenv import load_dotenv

# ───────────────────────────────────────────────────────
# 📊 Data Manipulation
# ───────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ───────────────────────────────────────────────────────
# 📈 Visualization
# ───────────────────────────────────────────────────────
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────
# 🧠 AI & LangChain
# ───────────────────────────────────────────────────────
from langchain.prompts import PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain, Neo4jVector
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# ───────────────────────────────────────────────────────
# 🧵 Streamlit App
# ───────────────────────────────────────────────────────
import streamlit as st

# ───────────────────────────────────────────────────────
# 🔌 Neo4j Driver
# ───────────────────────────────────────────────────────
from neo4j import GraphDatabase


def extract_features_from_sample_battery_from_text(file_text: str):
    # Load the q_d_n values from the text, handling comma separation per line.
    q_d_n_values = []
    for line in file_text.splitlines():
        # Remove leading/trailing whitespace and trailing commas
        value_str = line.strip().rstrip(',')
        if value_str:
            q_d_n_values.append(float(value_str))
    
    # Convert the list to a numpy array
    q_d_n_array = np.array(q_d_n_values)
    
    # Trim trailing zeros from the q_d_n array (assumes zeros at the end indicate no data)
    trimmed_q_d_n = np.trim_zeros(q_d_n_array, 'b')
    
    # Compute total cycles as the length of the trimmed array
    total_cycles = len(trimmed_q_d_n)

    # Define k values for which features are computed
    k_values = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    # Initialize the dictionary to hold features
    features = {}

    for k in k_values:
        # --- Last k cycles ---
        if total_cycles > k:
            slope_last = (trimmed_q_d_n[-1] - trimmed_q_d_n[-k]) / k
            grad_last = np.gradient(trimmed_q_d_n[-k:], 1)
            mean_grad_last = float(np.mean(grad_last))
        else:
            slope_last = np.nan
            mean_grad_last = np.nan
        
        features[f'slope_last_{k}_cycles'] = slope_last
        features[f'mean_grad_last_{k}_cycles'] = mean_grad_last

        # --- First k cycles ---
        if total_cycles > k:
            slope_first = (trimmed_q_d_n[k-1] - trimmed_q_d_n[0]) / k
            grad_first = np.gradient(trimmed_q_d_n[:k], 1)
            mean_grad_first = float(np.mean(grad_first))
        else:
            slope_first = np.nan
            mean_grad_first = np.nan

        features[f'slope_first_{k}_cycles'] = slope_first
        features[f'mean_grad_first_{k}_cycles'] = mean_grad_first

    # Add total cycles to the feature set
    features['total_cycles'] = total_cycles

    return features


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
    pre_delete_collection=False
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
    {
        "question": "Which batteries have similar mean_grad_first_500_cycles?",
        "query": "MATCH (b:Battery) WHERE abs(b.mean_grad_first_500_cycles - (-0.000578)) < 0.0001 RETURN b.battery_id, b.mean_grad_first_500_cycles"
    },
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embedder,
    vectorstore,
    k=3,
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

uploaded_file = st.file_uploader("📄 Upload a battery data file (.txt)", type=["txt"])
file_schema = None

if uploaded_file is not None:
    file_content = uploaded_file.read().decode("utf-8")
    try:
        features = extract_features_from_sample_battery_from_text(file_content)
        file_schema = "\n".join(f"{key}: {value}" for key, value in features.items())
        st.success("✅ File uploaded and processed successfully!")
        
        with st.expander("📊 Click to see extracted features"):
            sorted_features = dict(sorted(features.items()))
            for key, value in sorted_features.items():
                st.write(f"**{key}**: {value}")


        # --- Clean and parse q_d_n values for plotting ---
        q_values = []
        for line in file_content.strip().splitlines():
            val = line.strip().rstrip(',')
            if val:
                q_values.append(float(val))

        trimmed = np.trim_zeros(np.array(q_values), 'b')
        cycles = list(range(1, len(trimmed) + 1))

        # --- Plot ---
        fig, ax = plt.subplots()
        ax.plot(cycles, trimmed, label="Uploaded Battery")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("q_d_n")
        ax.set_title("Battery Degradation Over Cycles")
        ax.legend()
        st.pyplot(fig)

    except ValueError as e:
        st.error(f"⚠️ Error reading file: {e}")



def extract_battery_ids(response_text):
    return re.findall(r'\b(b\d+c\d+)\b', response_text)

import pandas as pd

def load_battery_data(battery_id):
    try:
        df = pd.read_csv(
            f"resources/raw/{battery_id}.txt", 
            header=None,  
            names=['capacity'],  
            engine='python',
            usecols=[0]
        )
        df.dropna(inplace=True)

        # --- Trim trailing zeroes ---
        trimmed = np.trim_zeros(df["capacity"].values, 'b')
        cycles = np.arange(1, len(trimmed) + 1)

        # --- Return cleaned DataFrame ---
        return pd.DataFrame({
            "cycle": cycles,
            "capacity": trimmed
        })

    except Exception as e:
        st.warning(f"Error loading battery {battery_id}: {e}")
        return None

import matplotlib.pyplot as plt

def plot_batteries(user_df, similar_ids):
    fig, ax = plt.subplots()
    if user_df is not None:
        ax.plot(user_df['cycle'], user_df['capacity'], label="Uploaded Battery", linewidth=2, color='black')

    for bid in similar_ids:
        df = load_battery_data(bid)
        if df is not None:
            ax.plot(df['cycle'], df['capacity'], label=bid, alpha=0.6)
            print(df['capacity'])

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity")
    ax.set_title("Battery Degradation Comparison")
    ax.legend(loc="best", fontsize="small")
    st.pyplot(fig)




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
    
    # --- Extract battery IDs ---
    battery_ids = extract_battery_ids(response.get("result", ""))
    print("Extracted similar battery: ",battery_ids)

    # --- Load uploaded battery data (from previous uploaded_file section) ---
    user_df = None
    if uploaded_file:
        try:
            q_values = []
            for line in file_content.strip().splitlines():
                val = line.strip().rstrip(',')
                try:
                    q_values.append(float(val))
                except ValueError:
                    continue
            trimmed = np.trim_zeros(np.array(q_values), 'b')
            user_df = pd.DataFrame({
                'cycle': np.arange(1, len(trimmed) + 1),
                'capacity': trimmed
            })
        except Exception as e:
            st.warning(f"Could not load uploaded battery for plotting: {e}")

    # --- Show AI text response ---
    st.subheader("🔎 AI Response:")
    st.write(response)
    
    # --- Plot both uploaded + retrieved batteries ---
    plot_batteries(user_df=user_df, similar_ids=battery_ids)


