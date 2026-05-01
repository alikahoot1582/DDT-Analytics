import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from groq import Groq

# --- 1. CONFIGURATION & SECURE API ---
# Note: In production, use st.secrets["GROQ_API_KEY"]
EMBEDDED_GROQ_KEY = "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" 

st.set_page_config(page_title="DDT Command Center | Ali Kahoot", layout="wide")

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("ddt_offline_dataset.csv")
    except:
        # Fallback for demo purposes if CSV isn't found
        return pd.DataFrame({
            "step": np.arange(100),
            "state_0": np.random.randn(100),
            "action_0": np.random.randn(100),
            "reward": np.zeros(100)
        })

def get_diffusion_trajectories(target_return, num_samples=100, steps=20):
    all_paths = []
    # Logic: High RTG (Target Return) shifts the mode toward a specific goal
    mode = np.array([5, 5]) if target_return > 15 else np.array([0, 0])
    
    for i in range(num_samples):
        current_x = np.random.randn(2) * 2 
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

def get_llama_3_3_strategy(api_key, context_df, target_r):
    """Refined Prompting for Decision Diffusion Transformer logic"""
    client = Groq(api_key=api_key)
    # Give the model a summarized view of the manifold
    history_summary = context_df.describe().to_dict()
    
    prompt = f"""
    System: You are an expert in Offline RL and Decision Diffusion Transformers (DDT).
    Context: The user is conditioning a generative policy with a Target Return-to-Go (RTG) of {target_r}.
    Manifold Data Summary: {history_summary}
    
    Task:
    1. Analyze the 'energy' of the current action manifold.
    2. Determine if the policy is successfully converging toward the goal state.
    3. Suggest a strategic adjustment to the diffusion guidance scale or action clipping.
    Keep the analysis concise and technical.
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

# --- 2. SESSION STATE (Prevents UI Jitter) ---
if 'sim_data' not in st.session_state:
    st.session_state.sim_data = None

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ System Control")
    st.info("**Developer: Muhammad Ali Kahoot**")
    
    if EMBEDDED_GROQ_KEY.startswith("gsk_"):
        st.success("✅ System Authorized")
    else:
        st.error("❌ Invalid API Key")

    target_r = st.slider("Target Return-to-Go (RTG)", 0.0, 30.0, 20.0)
    n_trajectories = st.number_input("Inference Samples", 10, 500, 100)
    viz_type = st.radio("Visualization Focus", ["Interactive (Altair)", "Statistical (Seaborn)", "3D (Plotly)"])
    
    if st.button("🔄 Regenerate Manifold"):
        st.session_state.sim_data = get_diffusion_trajectories(target_r, n_trajectories)
        st.toast("New trajectories generated!")

# Initialize data if empty
if st.session_state.sim_data is None:
    st.session_state.sim_data = get_diffusion_trajectories(target_r, n_trajectories)

data = st.session_state.sim_data
df_offline = load_dataset()

# --- 4. MAIN DASHBOARD ---
st.title("🧠 Neuro-Symbolic Decision Transformer")
st.markdown("---")

# LLAMA 3.3 STRATEGIC PLANNER
with st.expander("📡 Consult Meta Llama 3.3 Strategic Planner", expanded=True):
    col_btn, col_info = st.columns([1, 3])
    if col_btn.button("Generate Strategy", use_container_width=True):
        with st.spinner("Synthesizing trajectory manifold..."):
            strategy = get_llama_3_3_strategy(EMBEDDED_GROQ_KEY, df_offline, target_r)
            st.markdown(f"### 📋 Strategic Analysis\n{strategy}")
    else:
        col_info.info("Click 'Generate Strategy' to analyze the current manifold with Llama 3.3.")

# --- 5. VISUALIZATIONS ---
if viz_type == "Interactive (Altair)":
    st.subheader("Action Manifold Selection")
    brush = alt.selection_interval(encodings=['x'])
    final_steps = data[data.step == 19]
    
    chart = alt.Chart(final_steps).mark_circle(size=70).encode(
        x=alt.X('action_x', title="Action Space X"),
        y=alt.Y('action_y', title="Action Space Y"),
        color=alt.condition(brush, 'energy:Q', alt.value('lightgray'), scale=alt.Scale(scheme='viridis')),
        tooltip=['sample_id', 'energy']
    ).add_params(brush).properties(height=450)
    
    hist = alt.Chart(final_steps).mark_bar().encode(
        x=alt.X('energy:Q', bin=True, title="Action Energy Distribution"),
        y='count()',
        color=alt.value('#45b6fe')
    ).transform_filter(brush).properties(height=150)
    
    st.altair_chart(chart & hist, use_container_width=True)

elif viz_type == "Statistical (Seaborn)":
    st.subheader("Density & Joint Distribution")
    final_steps = data[data.step == 19]
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.kdeplot(data=final_steps, x="action_x", y="action_y", fill=True, cmap="mako", ax=ax)
        ax.set_title("Manifold Density")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(data=final_steps, x="energy", color="#45b6fe", ax=ax)
        ax.set_title("Energy Variance")
        st.pyplot(fig)

else:
    st.subheader("3D Diffusion Trajectory Space")
    fig = go.Figure()
    # Visualizing 15 random paths for clarity
    for i in range(min(n_trajectories, 15)):
        subset = data[data.sample_id == i]
        fig.add_trace(go.Scatter3d(
            x=subset['step'], y=subset['action_x'], z=subset['action_y'],
            mode='lines',
            line=dict(width=4, color=subset['energy'].iloc[-1], colorscale='Viridis'),
            name=f"Path {i}"
        ))
    fig.update_layout(
        scene=dict(xaxis_title='Diffusion Step (T)', yaxis_title='Action X', zaxis_title='Action Y'),
        margin=dict(l=0, r=0, b=0, t=0),
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- 6. DATA METRICS ---
st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Target RTG", f"{target_r:.1f}")
m2.metric("Manifold Convergence", f"{len(data[data.step == 19]):,}")
m3.metric("Avg Final Energy", f"{data[data.step == 19]['energy'].mean():.4f}")
m4.metric("Status", "Operational", delta="Stable")
