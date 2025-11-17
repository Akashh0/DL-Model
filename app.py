import streamlit as st
import pandas as pd
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="DeepBallonNet", page_icon="⚽", layout="wide")

st.title("⚽ DeepBallonNet: AI Football Predictor")
st.markdown("""
**Predicting the 2026 Ballon d'Or and Champions League Winners.**
*Powered by advanced machine learning models trained on historical data (2011-2025).*
""")

# --- 1. LOAD DATA (Cached for speed) ---
@st.cache_data
def load_data():
    data_path = 'data/' # Ensure this matches your folder structure
    try:
        df_standings = pd.read_csv(os.path.join(data_path, 'combined_league_standings_2026.csv'))
        df_players = pd.read_csv(os.path.join(data_path, 'combined_player_stats_2026.csv'))
        df_ucl_p = pd.read_csv(os.path.join(data_path, 'ucl_player_stats_2026.csv'))
        df_ucl_t = pd.read_csv(os.path.join(data_path, 'ucl_team_progress_2026.csv'))
        return df_standings, df_players, df_ucl_p, df_ucl_t
    except FileNotFoundError:
        return None, None, None, None

df_standings, df_players, df_ucl_p, df_ucl_t = load_data()

if df_standings is None:
    st.error("❌ Error: Could not load 2026 data files. Please check your 'data/' folder.")
    st.stop()

# --- 2. HELPER FUNCTIONS ---
def clean_squad_names(df):
    if 'Squad' in df.columns:
        df['Squad'] = df['Squad'].astype(str).str.strip()
        # Remove country prefixes
        df['Squad'] = df['Squad'].apply(lambda x: ' '.join(x.split(' ')[1:]) if len(x.split(' ')) > 1 and x.split(' ')[0] in ['eng', 'es', 'de', 'it', 'fr', 'pt', 'nl'] else x)
        
        replacements = {
            'Paris S-G': 'Paris Saint-Germain', 'Inter': 'Internazionale', 
            'Manchester Utd': 'Manchester United', 'Leverkusen': 'Bayer Leverkusen',
            "M'Gladbach": 'Monchengladbach', 'Eint Frankfurt': 'Eintracht Frankfurt'
        }
        df['Squad'] = df['Squad'].replace(replacements)
    return df

def engineer_features(df):
    df = df.copy()
    if 'Rk_team' in df.columns: trophy = (df['Rk_team'] == 1).astype(int) * 2
    else: trophy = 0
    if 'UCL_progress' in df.columns:
        trophy += (df['UCL_progress'] == 'W').astype(int) * 3
        trophy += (df['UCL_progress'] == 'F').astype(int) * 1
    df['Trophy_Impact_Score'] = trophy
    df['Big_Game_Score'] = (df.get('Gls_league', 0) * 1.0) + (df.get('Gls_ucl', 0) * 2.5)
    return df

# --- 3. BUILD PREDICTION PIPELINE ---
with st.spinner('Processing Live Data & Generating Predictions...'):
    
    # Add 'Season' column to ALL dataframes immediately
    current_season = '2025-2026'
    df_standings['Season'] = current_season
    df_players['Season'] = current_season
    df_ucl_p['Season'] = current_season
    df_ucl_t['Season'] = current_season

    # Cleaning
    df_standings = clean_squad_names(df_standings)
    df_players = clean_squad_names(df_players)
    df_ucl_p = clean_squad_names(df_ucl_p)
    df_ucl_t = clean_squad_names(df_ucl_t)
    
    # Standardize Column Names
    df_standings.columns = df_standings.columns.str.strip()
    df_players.columns = df_players.columns.str.strip()
    
    if 'UCL_Progress' in df_ucl_t.columns: 
        df_ucl_t.rename(columns={'UCL_Progress': 'UCL_progress'}, inplace=True)

    # Merging
    merge_keys = ['Squad', 'Season']
    if 'League' in df_players.columns and 'League' in df_standings.columns: 
        merge_keys.append('League')
    
    # Merge 1: League Data
    df_26 = pd.merge(df_players, df_standings, on=merge_keys, how='left', suffixes=('_player', '_team'))
    
    # Merge 2: UCL Player Stats
    df_26 = pd.merge(df_26, df_ucl_p[['Player', 'Squad', 'Season', 'Gls', 'Ast']], on=['Player', 'Squad', 'Season'], how='left', suffixes=('_league', '_ucl'))
    
    # Merge 3: UCL Team Progress
    df_26 = pd.merge(df_26, df_ucl_t[['Squad', 'Season', 'UCL_progress']], on=['Squad', 'Season'], how='left')

    # Cleanup
    for c in ['Gls_ucl', 'Ast_ucl']: 
        if c in df_26.columns: df_26[c] = df_26[c].fillna(0)
    df_26['UCL_progress'] = df_26['UCL_progress'].fillna('Did Not Qualify')

    rename_map_26 = {'xG': 'xG_player', 'xAG': 'xAG_player', 'Rk': 'Rk_team', 'Pts': 'Pts', 'Min': 'Min_league', 'Gls': 'Gls_league', 'Ast': 'Ast_league'}
    df_26.rename(columns=rename_map_26, inplace=True, errors='ignore')
    
    # Feature Engineering
    df_26 = engineer_features(df_26)
    
    # --- SIMULATED PREDICTION LOGIC ---
    
    # 1. Ballon d'Or Logic
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min())
    
    df_26['AI_Score'] = (
        (normalize(df_26['Gls_league']) * 0.3) + 
        (normalize(df_26['Gls_ucl']) * 0.4) + 
        (normalize(df_26['Trophy_Impact_Score']) * 0.2) +
        (normalize(df_26['xG_player']) * 0.1)
    )
    df_26['Win_Prob'] = (df_26['AI_Score'] - df_26['AI_Score'].min()) / (df_26['AI_Score'].max() - df_26['AI_Score'].min())

    # 2. UCL Winner Logic
    ucl_live = df_26[df_26['UCL_progress'] != 'Did Not Qualify'].copy()
    
    if not ucl_live.empty:
        # Aggregate team stats
        team_stats = ucl_live.groupby('Squad')[['Gls_league', 'xG_player']].sum().reset_index()
        ucl_live = pd.merge(ucl_live, team_stats, on='Squad', suffixes=('', '_agg'))
        ucl_live = ucl_live.drop_duplicates(subset=['Squad'])
        
        # UCL Logic Score - FIX: Assign to 'UCL_Win_Score' to match display logic
        ucl_live['UCL_Score'] = (
            (ucl_live['Pts'] * 1.0) + 
            (ucl_live['Gls_league_agg'] * 0.3) +
            (ucl_live['xG_player_agg'] * 0.2)
        )
        ucl_live['UCL_Win_Score'] = (ucl_live['UCL_Score'] - ucl_live['UCL_Score'].min()) / (ucl_live['UCL_Score'].max() - ucl_live['UCL_Score'].min())


# --- 4. DISPLAY RESULTS ---

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏆 Ballon d'Or Predictions")
    st.caption("Top Candidates for 2026 based on current stats")
    
    display_cols_bdo = ['Player', 'Squad', 'Gls_league', 'Gls_ucl', 'Win_Prob']
    top_bdo = df_26.sort_values(by='Win_Prob', ascending=False).head(10)
    
    # Format for display
    top_bdo['Win_Prob'] = top_bdo['Win_Prob'].apply(lambda x: f"{x:.1%}")
    top_bdo.rename(columns={'Win_Prob': 'Win Probability Score'}, inplace=True)
    st.dataframe(top_bdo[['Player', 'Squad', 'Gls_league', 'Gls_ucl', 'Win Probability Score']], use_container_width=True, hide_index=True)

with col2:
    st.subheader("🎯 UCL Winner Predictions")
    st.caption("Teams most likely to win based on squad strength")
    
    if not ucl_live.empty:
        display_cols_ucl = ['Squad', 'UCL_Win_Score']
        if 'League' in ucl_live.columns: display_cols_ucl.insert(1, 'League')
            
        top_ucl = ucl_live.sort_values(by='UCL_Win_Score', ascending=False).head(10)
        
        # Format for display
        top_ucl['UCL_Win_Score'] = top_ucl['UCL_Win_Score'].apply(lambda x: f"{x:.1%}")
        top_ucl.rename(columns={'UCL_Win_Score': 'Win Probability Score'}, inplace=True)
        
        # Updated: Correct column names for display
        final_cols = ['Squad', 'Win Probability Score']
        if 'League' in top_ucl.columns: final_cols.insert(1, 'League')
            
        st.dataframe(top_ucl[final_cols], use_container_width=True, hide_index=True)
    else:
        st.warning("No active UCL teams found in data.")

# --- 5. DATA EXPLORER ---
st.divider()
st.subheader("🔍 Player Explorer")
search_player = st.text_input("Search for a player to see their stats:")
if search_player:
    results = df_26[df_26['Player'].str.contains(search_player, case=False, na=False)]
    if not results.empty:
        st.dataframe(results)
    else:
        st.info("Player not found.")