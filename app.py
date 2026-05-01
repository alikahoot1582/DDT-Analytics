import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION & DATA LOADING ---
st.set_page_config(page_title="DDT Analytics | Muhammad Ali Kahoot", layout="wide")

@st.cache_data
def load_dataset():
    try:
        # Loading the dataset we generated earlier
        return pd.read_csv("ddt_offline_dataset.csv")
    except FileNotFoundError:
        return None

@st.cache_data
def get_inference_trajectories(target_rtg, num_samples=100, steps=20):
    """Simulates the reverse diffusion process based on user-defined RTG."""
    all_paths = []
    # Logic: High RTG targets expert mode (5,5), Low RTG targets origin (0,0)
    target_mode = np.array([5, 5]) if target_rtg > 15 else np.array([0, 0])
    
    for i in range(num_samples):
        curr_pos = np.random.randn(2) * 2  # Starting noise
        for t in range(steps):
            frac = t / steps
            noise = np.random.randn(2) * (1 - frac) * 0.4
            curr_pos = curr_pos * (1 - frac) + target_mode * frac + noise
            all_paths.append({
                'sample_id': i, 'step': t,
                'act_x': curr_pos[0], 'act_y': curr_pos[1],
                'energy': np.linalg.norm(curr_pos - target_mode)
            })
    return pd.DataFrame(all_paths)

# --- 2. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🎮 Dashboard Controls")
    target_rtg = st.slider("Target Return-to-Go", 0.0, 30.0, 15.0)
    samples = st.number_input("Inference Samples", 10, 500, 100)
    
    st.markdown("---")
    st.subheader("Visual Preferences")
    viz_choice = st.selectbox("Primary Chart", ["Interactive Altair", "Statistical KDE", "3D Plotly"])
    
    # --- SIGNATURE ---
    st.markdown("---")
    st.markdown("### 🛠️ Developer")
    st.info("**Made by Muhammad Ali Kahoot**")
    st.caption("Diffusion Decision Transformer Explorer v1.0")

# --- 3. MAIN INTERFACE ---
st.title("🧠 Diffusion-Based Decision Transformer")
st.markdown(f"**Targeting Reward Level:** `{target_rtg}`")

df_offline = load_dataset()
df_inference = get_inference_trajectories(target_rtg, samples)

tabs = st.tabs(["🚀 Model Inference", "📊 Dataset Analytics"])

# TABS 1: INFERENCE (THE MODEL IN ACTION)
with tabs[0]:
    if viz_choice == "Interactive Altair":
        brush = alt.selection_interval(encodings=['x'])
        points = alt.Chart(df_inference[df_inference.step == 19]).mark_circle(size=60).encode(
            x='act_x:Q', y='act_y:Q',
            color=alt.condition(brush, 'energy:Q', alt.value('lightgray'), scale=alt.Scale(scheme='magma')),
            tooltip=['energy']
        ).add_params(brush).properties(height=450)
        
        bars = alt.Chart(df_inference[df_inference.step == 19]).mark_bar().encode(
            x='count()', y=alt.Y('energy:Q', bin=True), color=alt.value('#ff4b4b')
        ).transform_filter(brush).properties(width=150)
        
        st.altair_chart(points | bars, use_container_width=True)

    elif viz_choice == "Statistical KDE":
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.kdeplot(data=df_inference[df_inference.step == 19], x="act_x", y="act_y", 
                    fill=True, cmap="Reds", ax=ax)
        ax.set_title("Action Probability Density")
        st.pyplot(fig)

    else:
        fig = go.Figure()
        for i in range(min(samples, 15)):
            sub = df_inference[df_inference.sample_id == i]
            fig.add_trace(go.Scatter3d(x=sub['step'], y=sub['act_x'], z=sub['act_y'], 
                                       mode='lines', line=dict(width=3)))
        fig.update_layout(scene=dict(xaxis_title='Step', yaxis_title='X', zaxis_title='Y'))
        st.plotly_chart(fig, use_container_width=True)

# TABS 2: DATASET (THE GROUND TRUTH)
with tabs[1]:
    if df_offline is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Offline Action Space**")
            st.scatter_chart(df_offline, x="action_0", y="action_1", color="return_to_go")
        with col2:
            st.write("**Reward Convergence**")
            st.line_chart(df_offline.groupby("step")["return_to_go"].mean())
    else:
        st.error("Please ensure 'ddt_offline_dataset.csv' is in the root directory.")

# --- 4. FOOTER ---
st.divider()
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Mean Energy", round(df_inference[df_inference.step == 19]['energy'].mean(), 4))
with c2:
    st.metric("Model Architecture", "DDT-L-12")
with c3:
    st.write("📝 **Author Statement:**")
    st.write("Generated via Muhammad Ali Kahoot's Research Lab.")
