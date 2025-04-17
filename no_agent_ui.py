import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ───────────────────────────────────────────────────────
# 🔐 Load Env
# ───────────────────────────────────────────────────────
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# ───────────────────────────────────────────────────────
# 📈 Feature Extraction
# ───────────────────────────────────────────────────────
def extract_slope_features(q_values):
    k_values = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    results = {}
    arr = np.trim_zeros(np.array(q_values), 'b')

    for k in k_values:
        if len(arr) >= k:
            grad = np.gradient(arr[-k:], 1)
            results[f"mean_grad_last_{k}_cycles"] = np.mean(grad)
            results[f"slope_last_{k}_cycles"] = (arr[-1] - arr[-k]) / k
            grad_first = np.gradient(arr[:k], 1)
            results[f"mean_grad_first_{k}_cycles"] = np.mean(grad_first)
            results[f"slope_first_{k}_cycles"] = (arr[k-1] - arr[0]) / k
        else:
            results[f"mean_grad_last_{k}_cycles"] = np.nan
            results[f"slope_last_{k}_cycles"] = np.nan
            results[f"mean_grad_first_{k}_cycles"] = np.nan
            results[f"slope_first_{k}_cycles"] = np.nan

    results["total_cycles"] = len(arr)
    return results

# ───────────────────────────────────────────────────────
# 🔍 Neo4j Query Function
# ───────────────────────────────────────────────────────
def query_similar_batteries(feature_name, feature_value, threshold=10, top_k=3, scale=1e6):
    with driver.session() as session:
        query = f"""
        MATCH (cp:ChargingPolicy)-[:USED_BY]->(b:Battery)
        WHERE abs(b.{feature_name} - $value) < $thresh
        RETURN b.battery_id AS battery_id,
               b.{feature_name} AS feature_value,
               abs(b.{feature_name} - $value) AS similarity,
               cp.charging_policy AS charging_policy
        ORDER BY similarity ASC
        LIMIT $topk
        """
        result = session.run(query, value=feature_value, thresh=threshold, topk=top_k)
        return [{
            "battery_id": r["battery_id"],
            "feature_value": r["feature_value"],
            "similarity": r["similarity"] * scale,
            "charging_policy": r["charging_policy"] or "Unknown"
        } for r in result]

# ───────────────────────────────────────────────────────
# 📁 Battery Data Loader
# ───────────────────────────────────────────────────────
def load_battery_curve(battery_id):
    try:
        df = pd.read_csv(
            f"resources/raw/{battery_id}.txt",
            header=None,
            names=['capacity'],
            engine='python',
            usecols=[0]
        )
        df.dropna(inplace=True)
        trimmed = np.trim_zeros(df["capacity"].values, 'b')
        cycles = np.arange(1, len(trimmed) + 1)
        return pd.DataFrame({"cycle": cycles, "capacity": trimmed})
    except Exception as e:
        st.warning(f"Error loading battery {battery_id}: {e}")
        return None

# ───────────────────────────────────────────────────────
# 🖥️ Streamlit UI
# ───────────────────────────────────────────────────────
st.title("🔋 Battery Similarity Finder (No AI)")

uploaded_file = st.file_uploader("📄 Upload battery degradation data (.txt)", type=["txt"])

# Feature type selection
feature_type = st.selectbox("Select Feature Type:", [
    "mean_grad_last",
    "mean_grad_first",
    "slope_last",
    "slope_first"
])

# k-window selection if needed
if feature_type != "total_cycles":
    selected_k = st.selectbox("Select Slope Window (k cycles):", [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], index=6)
    feature_key = f"{feature_type}_{selected_k}_cycles"
else:
    selected_k = None
    feature_key = "total_cycles"

# Similarity settings
scaled_threshold = st.number_input("🔧 Similarity Threshold (scaled)", min_value=1, max_value=5000, value=10, step=10)
actual_threshold = scaled_threshold / 1e6
top_k = st.slider("Top-K Results:", 1, 10, 3)

# ───────────────────────────────────────────────────────
# 📊 Process & Plot
# ───────────────────────────────────────────────────────
if uploaded_file:
    file_content = uploaded_file.read().decode("utf-8")
    try:
        q_values = [float(line.strip().replace(',', '')) for line in file_content.strip().splitlines()]
        features = extract_slope_features(q_values)

        # Get target value
        if feature_key == "total_cycles":
            slope_val = features["total_cycles"]
        else:
            slope_val = features[feature_key]

        st.success("✅ File processed!")
        st.write(f"📉 Extracted: **{feature_key} = {slope_val:.6f}**")

        # Neo4j retrieval
        results = query_similar_batteries(feature_key, slope_val, actual_threshold, top_k)

        if results:
            st.subheader("🔍 Similar Batteries")
            df = pd.DataFrame(results)
            st.dataframe(df)

            # Plot
            st.subheader("📊 Degradation Comparison Plot")
            fig, ax = plt.subplots()
            trimmed = np.trim_zeros(np.array(q_values), 'b')
            ax.plot(range(1, len(trimmed)+1), trimmed, label="Uploaded Battery", linewidth=2, color="black")

            for r in results:
                battery_df = load_battery_curve(r["battery_id"])
                if battery_df is not None:
                    ax.plot(battery_df["cycle"], battery_df["capacity"], label=r["battery_id"], alpha=0.6)

            ax.set_xlabel("Cycle")
            ax.set_ylabel("Capacity")
            ax.set_title("Battery Degradation Over Time")
            ax.legend(loc="best", fontsize="small")
            st.pyplot(fig)

        else:
            st.warning("No similar batteries found.")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
