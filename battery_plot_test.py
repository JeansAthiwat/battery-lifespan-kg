import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.title("🔋 Battery Degradation Plot Tester")

# Get available battery files
battery_dir = "resources/raw"
battery_files = [f.replace(".txt", "") for f in os.listdir(battery_dir) if f.endswith(".txt")]
selected_ids = st.multiselect("Select battery IDs to plot", options=sorted(battery_files))

def load_battery_data(battery_id):
    try:
        path = os.path.join(battery_dir, f"{battery_id}.txt")
        df = pd.read_csv(
            path,
            header=None,
            usecols=[0],  # Only use the first column
            names=["capacity"],
            engine="python"
        )
        df.dropna(inplace=True)
        df["cycle"] = np.arange(1, len(df) + 1)
        return df
    except Exception as e:
        st.warning(f"Failed to load {battery_id}: {e}")
        return None


if selected_ids:
    fig, ax = plt.subplots()
    for bid in selected_ids:
        df = load_battery_data(bid)
        if df is not None:
            ax.plot(df["cycle"], df["capacity"], label=bid, alpha=0.7)

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity")
    ax.set_title("Battery Degradation Comparison")
    ax.legend()
    st.pyplot(fig)
