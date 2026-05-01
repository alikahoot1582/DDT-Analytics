import streamlit as st
import pandas as pd
import numpy as np
import torch
import altair as alt
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from groq import Groq

# --- 1. CONFIGURATION & EMBEDDED API ---
# Replace 'your_api_key_here' with your actual Groq API key
EMBEDDED_GROQ_KEY = "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" 

st.set_page_config(page_title="DDT Command Center | Ali Kahoot", layout="wide")

@st.cache_data
def load_dataset():
    try:
        # Creating a dummy dataframe if the file is missing to keep the app functional
        return pd.read_csv("ddt_offline_dataset.csv")
    except:
        return pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])

@st.cache_data
def get_diffusion_trajectories(target_return, num_samples=100, steps=20):
    all_paths = []
    for i in range(num_samples):
        current_x = np.random.randn(2) * 2 
        mode = np.array([5, 5]) if target_return > 15 else np.array([0, 0])
        for t in range(steps):
            frac = t / steps
            noise = np.random.randn(2) * (1 - frac) * 0.5
            current_x = current_x * (1 - frac) + mode * frac + noise
            all_paths.append({
                'sample_id': i, 'step': t,
                'action_x': current_x[0], 'action_y': current_x[1],
                'energy': np.linalg.norm(current_x - mode)
            })
    return pd.DataFrame(all_paths)

def get_llama_3_1_strategy(api_key, context_df, target_r):
    client = Groq(api_key=api_key)
    history = context_df.tail(5).to_dict()
    prompt = f"Analyze this DDT trajectory: {history}. Goal Reward: {target_r}. What is the optimal action strategy?"
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ System Control")
    st.info("**Made by Muhammad Ali Kahoot**")
    
    # Visual confirmation that the key is embedded
    if EMBEDDED_GROQ_KEY != "your_api_key_here":
        st.success("✅ Groq API Key Embedded")
    else:
        st.error("⚠️ Replace 'your_api_key_here' in the code.")

    target_r = st.slider("Target Return-to-Go", 0.0, 30.0, 20.0)
    n_trajectories = st.number_input("Inference Samples", 10, 500, 100)
    viz_type = st.radio("Primary Focus", ["Interactive (Altair)", "Statistical (Seaborn)", "3D (Plotly)"])
    st.divider()
    st.caption("Architecture: Meta Llama 3.1 + Diffusion Head")

# --- 3. MAIN DASHBOARD ---
st.title("🧠 Neuro-Symbolic Decision Transformer")
df_offline = load_dataset()
data = get_diffusion_trajectories(target_r, num_samples=n_trajectories)

# --- LLAMA 3.1 REASONING SECTION ---
with st.expander("📡 Consult Meta Llama 3.1 Strategic Planner", expanded=True):
    if EMBEDDED_GROQ_KEY != "your_api_key_here":
        if st.button("Generate Strategy"):
            with st.spinner("Llama 3.1 is analyzing trajectories..."):
                strategy = get_llama_3_1_strategy(EMBEDDED_GROQ_KEY, df_offline, target_r)
                st.markdown(f"**Llama 3.1 Analysis:**\n\n{strategy}")
    else:
        st.warning("Please embed your API key in the code to enable reasoning.")

st.divider()

# --- 4. VISUALIZATIONS ---
# (Keeping your original visualization logic)
if viz_type == "Interactive (Altair)":
    st.subheader("Action Manifold Selection")
    brush = alt.selection_interval(encodings=['x'])
    chart = alt.Chart(data[data.step == 19]).mark_circle(size=60).encode(
        x='action_x', y='action_y',
        color=alt.condition(brush, 'energy:Q', alt.value('lightgray'), scale=alt.Scale(scheme='viridis')),
        tooltip=['sample_id', 'energy']
    ).add_params(brush).properties(height=400)
    hist = alt.Chart(data[data.step == 19]).mark_bar().encode(
        x=alt.X('energy:Q', bin=True), y='count()', color=alt.value('#45b6fe')
    ).transform_filter(brush).properties(height=200)
    st.altair_chart(chart & hist, use_container_width=True)

elif viz_type == "Statistical (Seaborn)":
    st.subheader("Density & Joint Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.kdeplot(data=data[data.step == 19], x="action_x", y="action_y", fill=True, cmap="mako", ax=ax)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(data=data[data.step == 19], x="energy", palette="Set2", ax=ax)
        st.pyplot(fig)

else:
    st.subheader("3D Diffusion Trajectory Space")
    fig = go.Figure()
    for i in range(min(n_trajectories, 15)):
        subset = data[data.sample_id == i]
        fig.add_trace(go.Scatter3d(x=subset['step'], y=subset['action_x'], z=subset['action_y'],
                                   mode='lines', line=dict(width=3, color=subset['energy'].iloc[-1])))
    fig.update_layout(scene=dict(xaxis_title='T', yaxis_title='Act X', zaxis_title='Act Y'), template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# --- 5. DATA METRICS ---
st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Target RTG", target_r)
c2.metric("Mean Final Energy", round(data[data.step == 19]['energy'].mean(), 4))
c3.metric("Developer", "Ali Kahoot")
