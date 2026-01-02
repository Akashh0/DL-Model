import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ProphetPlay", 
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1 { text-align: center; color: #4EA8DE; font-family: 'Helvetica Neue', sans-serif; }
    .stDataFrame { border: 1px solid #303030; border-radius: 5px; }
    .editor-text { font-family: 'Georgia', serif; color: #e0e0e0; line-height: 1.6; font-size: 1.1em; background-color: #1f1f1f; padding: 20px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("⚽ ProphetPlay: The Journalist AI")
st.markdown("""
<div style='text-align: center; color: #aaaaaa; margin-bottom: 30px;'>
    <b>Predicting the 2026 Ballon d'Or and Champions League Winners.</b><br>
    <i>Powered by "Journalist View" Analytics • Heritage Bonus • Scenario Simulation</i>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. SETUP & CONFIG
# ==============================================================================

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("The AI uses a default weighted model (Narrative 30%, Stats 40%). Use the tools below to tweak scenarios.")
    
    st.divider()
    st.header("🔮 Scenario Simulator")
    force_ucl_winner = st.selectbox(
        "🏆 Force UCL Winner:",
        ["None (Use Live Data)", "Real Madrid", "Bayern Munich", "Manchester City", "Arsenal", "Barcelona", "Paris Saint-Germain"]
    )

@st.cache_data
def load_data():
    data_path = 'data/'
    try:
        # Load Current 2026 Data
        try:
            df = pd.read_csv(os.path.join(data_path, 'master_dataset_2026.csv'), encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(os.path.join(data_path, 'master_dataset_2026.csv'), encoding='latin1')
        return df
    except FileNotFoundError:
        return None

def fix_mojibake(text):
    if not isinstance(text, str): return text
    replacements = { 'Du\x9a': 'Duš', 'Duš': 'Duš', 'Vlahovi': 'Vlahović', 'MbappÃ©': 'Mbappé', 'Ã': 'í', 'AtlÃ©tico': 'Atlético' }
    for bad, good in replacements.items():
        if bad in text: text = text.replace(bad, good)
    try: return text.encode('latin-1').decode('utf-8')
    except: return text

def safe_rename(df):
    target_map = { 'xG': 'xG_player', 'xAG': 'xAG_player', 'Rk': 'Rk_team', 'Pts': 'Pts', 'Min': 'Min_league', 'Gls': 'Gls_league', 'Ast': 'Ast_league', 'UCL_Progress': 'UCL_progress' }
    clean_map = {s: t for s, t in target_map.items() if s in df.columns and t not in df.columns}
    if clean_map: df.rename(columns=clean_map, inplace=True)
    return df

# ==============================================================================
# 2. ANALYTICS ENGINES
# ==============================================================================

def run_ballon_dor_engine(df, forced_winner="None (Use Live Data)", weights=None):
    df = df.copy()
    
    # Default weights if not provided
    if weights is None:
        weights = {'stats': 40, 'narrative': 30, 'bias': 15, 'ucl': 15}

    # --- SCENARIO LOGIC ---
    if forced_winner != "None (Use Live Data)":
        # Overwrite UCL progress for the selected team
        df.loc[df['Squad'] == forced_winner, 'UCL_progress'] = 'W'
        # Downgrade others who might have 'W' in live data
        df.loc[df['Squad'] != forced_winner, 'UCL_progress'] = df.loc[df['Squad'] != forced_winner, 'UCL_progress'].replace('W', 'SF')

    # Feature Engineering
    df['Total_GA'] = df.get('Gls_league', 0) + df.get('Ast_league', 0) + df.get('Gls_ucl', 0) + df.get('Ast_ucl', 0)
    
    # Narrative Score
    trophy_score = 0
    if 'Rk_team' in df.columns:
        rank = pd.to_numeric(df['Rk_team'], errors='coerce').fillna(10)
        trophy_score += (rank == 1).astype(int) * 5
    
    if 'UCL_progress' in df.columns:
        ucl = df['UCL_progress'].astype(str).str.strip()
        trophy_score += (ucl == 'W').astype(int) * 10 
        trophy_score += (ucl == 'F').astype(int) * 5 
    
    df['Narrative_Score'] = trophy_score

    # Media Bias
    media_darlings = ['Real Madrid', 'Barcelona', 'Manchester City', 'Bayern Munich', 'Liverpool', 'Paris S-G', 'Paris Saint-Germain']
    df['Media_Bias'] = 0
    if 'Squad' in df.columns:
        for club in media_darlings:
            df.loc[df['Squad'].astype(str).str.contains(club, case=False, na=False), 'Media_Bias'] = 1

    # Scoring
    def norm(s): return (s - s.min()) / (s.max() - s.min())
    
    # Handle NaNs
    df['Total_GA'] = df['Total_GA'].fillna(0)
    
    ga_norm = norm(df['Total_GA'])
    narrative_norm = norm(df['Narrative_Score'])
    
    df['Journalist_Points'] = (
        (ga_norm * weights['stats']) + 
        (narrative_norm * weights['narrative']) + 
        (df['Media_Bias'] * weights['bias']) + 
        (norm(df.get('Gls_ucl', 0)) * weights['ucl'])
    )
    
    df['Power Index'] = (df['Journalist_Points'] / df['Journalist_Points'].max()) * 99.0
    return df.sort_values(by='Power Index', ascending=False).head(15)

def run_ucl_engine(df, forced_winner="None (Use Live Data)"):
    # Aggregate
    agg_rules = {
        'Rk_team': 'min', 'Pts': 'max', 'GF': 'max', 'GA': 'max',
        'W': 'max', 'D': 'max', 'L': 'max', 'Gls_ucl': 'sum'
    }
    valid_rules = {k:v for k,v in agg_rules.items() if k in df.columns}
    team_df = df.groupby('Squad').agg(valid_rules).reset_index()
    
    # Engineer
    team_df['Attack_Power'] = (team_df.get('GF', 0) * 0.6) + (team_df.get('Gls_ucl', 0) * 2.0)
    matches = team_df.get('W', 0) + team_df.get('D', 0) + team_df.get('L', 1).replace(0, 1)
    team_df['Win_Rate'] = team_df.get('W', 0) / matches
    
    heritage_boost = ['Real Madrid', 'Bayern Munich', 'Liverpool', 'AC Milan', 'Barcelona', 'Manchester City']
    team_df['Heritage_Bonus'] = team_df['Squad'].apply(lambda x: 15 if x in heritage_boost else 0)
    
    # Scoring
    def norm(s): return (s - s.min()) / (s.max() - s.min())

    team_df['Score'] = (
        (norm(team_df['Attack_Power']) * 40) + (norm(team_df['Win_Rate']) * 30) + (team_df['Heritage_Bonus'])
    )
    
    # --- SCENARIO OVERRIDE ---
    if forced_winner != "None (Use Live Data)":
        # Give the forced winner massive points to push them to #1
        team_df.loc[team_df['Squad'] == forced_winner, 'Score'] += 100

    team_df['Power Index'] = (team_df['Score'] / team_df['Score'].max()) * 99.0
    scores = np.exp(team_df['Power Index'] / 15)
    team_df['Title Odds'] = scores / scores.sum()
    
    return team_df.sort_values(by='Power Index', ascending=False).head(10)

# --- FEATURE 1: RADAR CHART (Restored) ---
def plot_radar(player1, player2, df):
    p1_data = df[df['Player'] == player1].iloc[0]
    p2_data = df[df['Player'] == player2].iloc[0]
    
    categories = ['Total Goals', 'Narrative', 'Media Bias', 'UCL Goals', 'League Stats']
    
    def get_values(row):
        # Normalize for visualization (0-100)
        return [
            min((row['Total_GA'] / 50) * 100, 100),
            min((row['Narrative_Score'] / 15) * 100, 100),
            row['Media_Bias'] * 100,
            min((row['Gls_ucl'] / 12) * 100, 100),
            min((row['Gls_league'] / 35) * 100, 100)
        ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=get_values(p1_data), theta=categories, fill='toself', name=player1, line_color='#4EA8DE'))
    fig.add_trace(go.Scatterpolar(r=get_values(p2_data), theta=categories, fill='toself', name=player2, line_color='#FF4B4B'))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        margin=dict(l=40, r=40, t=20, b=20),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white")
    )
    return fig

# ==============================================================================
# 3. MAIN APP EXECUTION
# ==============================================================================

df = load_data()

if df is None:
    st.error("❌ **Critical Error:** Could not load `master_dataset_2026.csv`.")
    st.stop()

# --- Pipeline ---
with st.spinner("Running DeepBallonNet Engines..."):
    df = df.reset_index(drop=True).loc[:, ~df.columns.duplicated()]
    df = safe_rename(df)
    for col in ['Player', 'Squad']: 
        if col in df.columns: df[col] = df[col].apply(fix_mojibake)
            
    bdo_rankings = run_ballon_dor_engine(df, force_ucl_winner)
    ucl_rankings = run_ucl_engine(df, force_ucl_winner)

# --- LAYOUT ---
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("🏆 Ballon d'Or Rankings")
    display_bdo = bdo_rankings[['Player', 'Squad', 'Total_GA', 'Power Index']].copy()
    display_bdo.columns = ['Player', 'Club', 'G+A', 'Power Rating']
    display_bdo.index = range(1, len(display_bdo) + 1)
    
    st.dataframe(
        display_bdo,
        column_config={
            "Power Rating": st.column_config.ProgressColumn("AI Rating", format="%.1f", min_value=0, max_value=100),
            "G+A": st.column_config.NumberColumn("Total G/A")
        }, use_container_width=True, height=400
    )

with col2:
    st.subheader("🇪🇺 UCL Contenders")
    display_ucl = ucl_rankings[['Squad', 'Title Odds', 'Attack_Power', 'Power Index']].copy()
    display_ucl.columns = ['Club', 'Probability', 'Attack Rtg', 'Power Rating']
    display_ucl.index = range(1, len(display_ucl) + 1)
    
    st.dataframe(
        display_ucl,
        column_config={
            "Power Rating": st.column_config.ProgressColumn("Power Index", format="%.1f", min_value=0, max_value=100),
            "Probability": st.column_config.NumberColumn("Odds", format="%.1%")
        }, use_container_width=True, height=400
    )

# --- ANALYSIS SECTION ---
st.divider()

col_radar, col_controls = st.columns([1, 1.5], gap="large")

with col_radar:
    st.subheader("⚔️ Head-to-Head")
    st.caption("Compare the stats behind the rank.")
    p1 = st.selectbox("Player A", bdo_rankings['Player'], index=0)
    p2 = st.selectbox("Player B", bdo_rankings['Player'], index=1)
    st.plotly_chart(plot_radar(p1, p2, bdo_rankings), use_container_width=True)

with col_controls:
    st.subheader("🎛️ Build Your Own Algorithm")
    st.caption("Don't agree with the AI? Adjust the weights to create your own criteria.")
    
    w_stats = st.slider("📊 Importance of Stats (Goals/Assists)", 0, 100, 40)
    w_narrative = st.slider("🏆 Importance of Trophies (Narrative)", 0, 100, 30)
    w_ucl = st.slider("🇪🇺 Importance of UCL Performance", 0, 100, 15)
    w_bias = st.slider("🌍 Importance of Club Heritage (Media Bias)", 0, 100, 15)
    
    if st.button("Apply Custom Weights 🔄"):
        custom_weights = {'stats': w_stats, 'narrative': w_narrative, 'bias': w_bias, 'ucl': w_ucl}
        custom_rankings = run_ballon_dor_engine(df, force_ucl_winner, custom_weights)
        
        st.success("Rankings Updated based on your criteria!")
        
        display_custom = custom_rankings[['Player', 'Squad', 'Total_GA', 'Power Index']].copy()
        display_custom.columns = ['Player', 'Club', 'G+A', 'Power Rating']
        display_custom.index = range(1, len(display_custom) + 1)
        
        st.dataframe(
            display_custom.head(10),
            column_config={
                "Power Rating": st.column_config.ProgressColumn("Custom Rating", format="%.1f", min_value=0, max_value=100),
            }, use_container_width=True
        )

# --- SCOUTING ---
with st.expander("🔎 Scout Player Database"):
    search_term = st.text_input("Find Player:")
    if search_term:
        res = df[df['Player'].astype(str).str.contains(search_term, case=False)]
        st.dataframe(res)