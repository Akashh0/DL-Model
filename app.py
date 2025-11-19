import streamlit as st
import pandas as pd
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="DeepBallonNet 2026", 
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        text-align: center; 
        color: #4EA8DE;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h3 {
        border-bottom: 2px solid #4EA8DE;
        padding-bottom: 10px;
    }
    .stDataFrame {
        border: 1px solid #303030;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚽ ProphetPlay: The Journalist AI")
st.markdown("""
<div style='text-align: center; color: #aaaaaa; margin-bottom: 30px;'>
    <b>Predicting the 2026 Ballon d'Or and Champions League Winners.</b><br>
    <i>Powered by "Journalist View" Analytics • Heritage Bonus • Narrative Scoring</i>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. DATA LOADING & CLEANING
# ==============================================================================
@st.cache_data
def load_data():
    data_path = 'data/' # Check your folder structure
    try:
        # Try loading with UTF-8 first, fallback to latin1 if needed
        try:
            df_master = pd.read_csv(os.path.join(data_path, 'master_dataset_2026.csv'), encoding='utf-8')
        except UnicodeDecodeError:
            df_master = pd.read_csv(os.path.join(data_path, 'master_dataset_2026.csv'), encoding='latin1')
            
        return df_master
    except FileNotFoundError:
        return None

def fix_mojibake(text):
    """Repairs broken text characters (e.g. MbappÃ© -> Mbappé)"""
    if not isinstance(text, str): return text
    replacements = {
        'Du\x9a': 'Duš', 'Duš': 'Duš', 'Vlahovi': 'Vlahović',
        'GyÃ¶keres': 'Gyökeres', 'Lewandowski': 'Lewandowski',
        'MbappÃ©': 'Mbappé', 'Ã': 'í', 'AtlÃ©tico': 'Atlético'
    }
    for bad, good in replacements.items():
        if bad in text: text = text.replace(bad, good)
    try:
        return text.encode('latin-1').decode('utf-8')
    except:
        return text

def safe_rename(df):
    target_map = {
        'xG': 'xG_player', 'xAG': 'xAG_player', 'Rk': 'Rk_team', 'Pts': 'Pts',
        'Min': 'Min_league', 'Gls': 'Gls_league', 'Ast': 'Ast_league',
        'UCL_Progress': 'UCL_progress'
    }
    clean_map = {s: t for s, t in target_map.items() if s in df.columns and t not in df.columns}
    if clean_map: df.rename(columns=clean_map, inplace=True)
    return df

# ==============================================================================
# 2. ANALYTICS ENGINES
# ==============================================================================

def run_ballon_dor_engine(df):
    """
    Simulates the XGBoost 'Journalist View' Model using weighted heuristics.
    """
    df = df.copy()
    
    # 1. Feature Engineering
    df['Total_GA'] = df.get('Gls_league', 0) + df.get('Ast_league', 0) + df.get('Gls_ucl', 0) + df.get('Ast_ucl', 0)
    
    # 2. Narrative Score (Trophies)
    trophy_score = 0
    if 'Rk_team' in df.columns:
        rank = pd.to_numeric(df['Rk_team'], errors='coerce').fillna(10)
        trophy_score += (rank == 1).astype(int) * 5  # League Winner
    
    if 'UCL_progress' in df.columns:
        ucl = df['UCL_progress'].astype(str).str.strip()
        trophy_score += (ucl == 'W').astype(int) * 10 # UCL Winner (Huge)
        trophy_score += (ucl == 'F').astype(int) * 5  # Finalist
    
    df['Narrative_Score'] = trophy_score

    # 3. Media Bias (The "Real Madrid Tax")
    media_darlings = ['Real Madrid', 'Barcelona', 'Manchester City', 'Bayern Munich', 'Liverpool', 'Paris S-G']
    df['Media_Bias'] = 0
    if 'Squad' in df.columns:
        for club in media_darlings:
            df.loc[df['Squad'].astype(str).str.contains(club, case=False, na=False), 'Media_Bias'] = 1

    # 4. Scoring Formula (Approximating the XGBoost Feature Importance)
    # Weights: Narrative (35%), Total G/A (35%), Media Bias (15%), Efficiency (15%)
    
    # Normalize
    def norm(s): return (s - s.min()) / (s.max() - s.min())
    
    # Handle edge case where max == min
    ga_norm = norm(df['Total_GA'].fillna(0))
    narrative_norm = norm(df['Narrative_Score'])
    
    df['Journalist_Points'] = (
        (ga_norm * 40) + 
        (narrative_norm * 30) + 
        (df['Media_Bias'] * 15) +
        (norm(df.get('Gls_ucl', 0)) * 15) # UCL Goals Bonus
    )
    
    # Final Polish
    df['Power Index'] = (df['Journalist_Points'] / df['Journalist_Points'].max()) * 99.0
    return df.sort_values(by='Power Index', ascending=False).head(10)

def run_ucl_engine(df):
    """
    Simulates the DeepUCLNet 'Power Ranking' Model using aggregated team stats.
    """
    # 1. Aggregate Player -> Team
    agg_rules = {
        'Rk_team': 'min', 'Pts': 'max', 'GF': 'max', 'GA': 'max',
        'W': 'max', 'D': 'max', 'L': 'max', 'Gls_ucl': 'sum'
    }
    valid_rules = {k:v for k,v in agg_rules.items() if k in df.columns}
    
    team_df = df.groupby('Squad').agg(valid_rules).reset_index()
    
    # 2. Engineer Features
    team_df['Attack_Power'] = (team_df.get('GF', 0) * 0.6) + (team_df.get('Gls_ucl', 0) * 2.0)
    
    matches = team_df.get('W', 0) + team_df.get('D', 0) + team_df.get('L', 1).replace(0, 1)
    team_df['Win_Rate'] = team_df.get('W', 0) / matches
    
    # 3. Heritage Bonus (The "DNA" Factor)
    heritage_boost = ['Real Madrid', 'Bayern Munich', 'Liverpool', 'AC Milan', 'Barcelona', 'Manchester City']
    team_df['Heritage_Bonus'] = team_df['Squad'].apply(lambda x: 15 if x in heritage_boost else 0)
    
    # 4. Scoring Formula
    # Weights: Attack (40%), Win Rate (30%), Heritage (30%)
    def norm(s): return (s - s.min()) / (s.max() - s.min())

    team_df['Score'] = (
        (norm(team_df['Attack_Power']) * 40) +
        (norm(team_df['Win_Rate']) * 30) +
        (team_df['Heritage_Bonus']) # Raw points added
    )
    
    # Scale to 99
    team_df['Power Index'] = (team_df['Score'] / team_df['Score'].max()) * 99.0
    
    # Calculate Title Odds (Softmax proxy)
    scores = np.exp(team_df['Power Index'] / 15) # Temperature scaling
    team_df['Title Odds'] = scores / scores.sum()
    
    return team_df.sort_values(by='Power Index', ascending=False).head(10)

# ==============================================================================
# 3. MAIN APP EXECUTION
# ==============================================================================

df = load_data()

if df is None:
    st.error("❌ **Critical Error:** Could not load `master_dataset_2026.csv`.")
    st.info("Please ensure the file is in the `data/` folder.")
    st.stop()

# --- Pipeline Execution ---
with st.spinner("running DeepBallonNet neural engines..."):
    # 1. Clean
    df = df.reset_index(drop=True)
    df = df.loc[:, ~df.columns.duplicated()]
    df = safe_rename(df)
    
    # 2. Text Repair
    for col in ['Player', 'Squad', 'Nation']:
        if col in df.columns:
            df[col] = df[col].apply(fix_mojibake)
            
    # 3. Run Engines
    bdo_rankings = run_ballon_dor_engine(df)
    ucl_rankings = run_ucl_engine(df)

# --- LAYOUT ---

col1, col2 = st.columns([1, 1], gap="medium")

# --- LEFT: BALLON D'OR ---
with col1:
    st.subheader("🏆 Ballon d'Or Power Rankings")
    st.caption("Top 15 candidates based on Stats, Narrative & Media Bias")
    
    # Format Table
    display_bdo = bdo_rankings[['Player', 'Squad', 'Total_GA', 'Power Index']].copy()
    display_bdo.columns = ['Player', 'Club', 'G+A', 'Power Rating']
    display_bdo = display_bdo.reset_index(drop=True)
    display_bdo.index += 1
    
    st.dataframe(
        display_bdo,
        column_config={
            "Power Rating": st.column_config.ProgressColumn(
                "AI Rating",
                help="The model's confidence score (0-100)",
                format="%.1f",
                min_value=0,
                max_value=100,
            ),
            "G+A": st.column_config.NumberColumn("Total G/A")
        },
        use_container_width=True
    )
    
    # Top Contender Highlight
    winner = display_bdo.iloc[0]
    st.success(f"**Projected Winner:** {winner['Player']} ({winner['Club']})")

# --- RIGHT: CHAMPIONS LEAGUE ---
with col2:
    st.subheader("UCL Title Contenders")
    st.caption("Top 10 Teams based on Attack Power, Form & Heritage")
    
    # Format Table
    display_ucl = ucl_rankings[['Squad', 'Title Odds', 'Attack_Power', 'Power Index']].copy()
    display_ucl.columns = ['Club', 'Probability', 'Attack Rtg', 'Power Rating']
    display_ucl = display_ucl.reset_index(drop=True)
    display_ucl.index += 1
    
    st.dataframe(
        display_ucl,
        column_config={
            "Power Rating": st.column_config.ProgressColumn(
                "Power Index",
                format="%.1f",
                min_value=0,
                max_value=100,
            ),
            "Probability": st.column_config.NumberColumn(
                "Odds",
                format="%.1%"
            ),
             "Attack Rtg": st.column_config.NumberColumn(
                "Atk Rtg",
                format="%.0f"
            )
        },
        use_container_width=True
    )
    
    # Top Contender Highlight
    winner_team = display_ucl.iloc[0]
    st.info(f"**Favorite:** {winner_team['Club']} ({winner_team['Probability']:.1%} chance)")

# --- EXPLORER SECTION ---
st.divider()
st.subheader("🔍 Scout Report")

with st.expander("Search Database"):
    search_term = st.text_input("Find Player or Team:", placeholder="e.g. Lamine Yamal")
    if search_term:
        res = df[df['Player'].astype(str).str.contains(search_term, case=False) | 
                 df['Squad'].astype(str).str.contains(search_term, case=False)]
        
        if not res.empty:
            st.dataframe(res[['Player', 'Squad', 'Age', 'Gls_league', 'Ast_league', 'Gls_ucl', 'Rk_team']])
        else:
            st.warning("No records found.")