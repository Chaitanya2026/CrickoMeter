import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime
import numpy as np

# === Page Config ===
st.set_page_config(page_title="CrickoMeter", page_icon="🏏", layout="wide")

# === Custom CSS ===
st.markdown("""
    <style>
        h1 {
            text-align: center;
            color: #0e1117;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .st-emotion-cache-1avcm0n {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
        }
        .stMetric {
            background-color: #f9f9f9;
            border-radius: 10px;
            padding: 10px;
        }
        .stButton > button {
            color: white;
            background: linear-gradient(to right, #4facfe, #00f2fe);
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# === App Title ===
st.markdown("<h1>🏏 CrickoMeter: Predict & Analyze Cricket Matches</h1>", unsafe_allow_html=True)

# === Load Models ===
score_model = joblib.load("models/score_model_1.pkl")
boundary_model = joblib.load("models/boundary_model_1.pkl")
bucket_model = joblib.load("models/bucket_model_1.pkl")
bucket_label_encoder = joblib.load("models/bucket_label_encoder_1.pkl")
win_model = joblib.load("models/win_model_1.pkl")
win_model_scaler = joblib.load("models/win_model_scaler_1.pkl")

# === Load Data ===
df = pd.read_csv("combined_3_dropdowns.csv")

# === Venue Fixes ===
venue_map = {
    "Dr. DY Patil Sports Academy": "Dr. DY Patil Sports Academy, Mumbai",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala": "Himachal Pradesh Cricket Association Stadium",
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur": "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh",
    "Punjab Cricket Association IS Bindra Stadium": "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh",
    "Maharashtra Cricket Association Stadium": "Maharashtra Cricket Association Stadium, Pune",
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium, Bengaluru",
    "MA Chidambaram Stadium , Chepauk": "MA Chidambaram Stadium, Chepauk, Chennai",
    "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh",
    "Punjab Crciket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
    "Civil Service Cricket Club, Stormont": "Civil Service Cricket Club, Stormont, Belfast",
    "County Ground": "County Ground, Bristol",
    "College Field": "College Field, St Peter Port",
    "Desert Springs Cricket Ground": "Desert Springs Cricket Ground, Almeria",
    "Eden Park": "Eden Park, Auckland",
    "Gahanga International Cricket Stadium. Rwanda": "Gahanga International Cricket Stadium, Rwanda",
    "Grange Cricket Club Ground, Raeburn Place": "Grange Cricket Club, Raeburn Place",
    "Grange Cricket Club Ground, Raeburn Place, Edinburgh": "Grange Cricket Club, Raeburn Place",
    "Hagley Oval": "Hagley Oval, Christchurch",
    "Holkar Cricket Stadium": "Holkar Cricket Stadium, Indore",
    "Indian Association Ground": "Indian Association Ground, Singapore",
    "Kennington Oval": "Kennington Oval, London",
    "Kensington Oval, Bridgetown": "Kensington Oval, Bridgetown, Barbados",
    "King George V Sports Ground": "King George V Sports Ground, Castel",
    "Kingsmead": "Kingsmead, Durban",
    "Manuka Oval": "Manuka Oval, Canberra",
    "Maple Leaf North-West Ground": "Maple Leaf North-West Ground, King City",
    "McLean Park": "McLean Park, Napier",
    "Mission Road Ground, Mong Kok": "Mission Road Ground, Mong Kok, Hong Kong",
    "Moara Vlasiei Cricket Ground, Ilfov County": "Moara Vlasiei Cricket Ground",
    "National Cricket Stadium, St George's, Grenada": "National Cricket Stadium, Grenada",
    "Old Trafford": "Old Trafford, Manchester",
    "Providence Stadium": "Providence Stadium, Guyana",
    "Queens Sports Club": "Queens Sports Club, Bulawayo",
    "Riverside Ground, Chester-le-Street": "Riverside Ground",
    "Sabina Park, Kingston": "Sabina Park, Kingston, Jamaica",
    "Saurashtra Cricket Association Stadium": "Saurashtra Cricket Association Stadium, Rajkot",
    "Saxton Oval": "Saxton Oval, Nelson",
    "Seddon Park": "Seddon Park, Hamilton",
    "Shere Bangla National Stadium": "Shere Bangla National Stadium, Mirpur",
    "Sky Stadium": "Sky Stadium, Wellington",
    "Sophia Gardens": "Sophia Gardens, Cardiff",
    "Sportpark Het Schootsveld, Deventer": "Sportpark Het Schootsveld",
    "Sportpark Maarschalkerweerd, Utrecht": "Sportpark Maarschalkerweerd",
    "Sportpark Westvliet, The Hague": "Sportpark Westvliet",
    "St George's Park": "St George's Park, Gqeberha",
    "SuperSport Park": "SuperSport Park, Centurion",
    "Terdthai Cricket Ground": "Terdthai Cricket Ground, Bangkok",
    "The Rose Bowl": "The Rose Bowl, Southampton",
    "The Village, Malahide": "The Village, Malahide, Dublin",
    "The Wanderers Stadium": "The Wanderers Stadium, Johannesburg",
    "Trent Bridge": "Trent Bridge, Nottingham",
    "Tribhuvan University International Cricket Ground, Kirtipur": "Tribhuvan University International Cricket Ground",
    "United Cricket Club Ground": "United Cricket Club Ground, Windhoek",
    "Vidarbha Cricket Association Stadium, Jamtha": "Vidarbha Cricket Association Stadium, Jamtha, Nagpur",
    "Wanderers Cricket Ground, Windhoek": "Wanderers Cricket Ground",
    "Wanderers": "Wanderers Cricket Ground",
    "Warner Park, Basseterre, St Kitts": "Warner Park, St Kitts",
    "Warner Park, Basseterre": "Warner Park, St Kitts",
    "Windsor Park, Roseau": "Windsor Park, Roseau, Dominica",
    "Central Broward Regional Park Stadium": "Central Broward Regional Park Stadium Turf Ground"
}
df['venue'] = df['venue'].replace(venue_map)

# === Drop legacy teams ===
old_teams = ['Pune Warriors', 'Deccan Chargers', 'Gujarat Lions',
             'Rising Pune Supergiant', 'Rising Pune Supergiants', 'Kochi Tuskers Kerala']
df = df[~df['innings_1_team'].isin(old_teams)]
df = df[~df['innings_2_team'].isin(old_teams)]

# === Sidebar Inputs ===
with st.sidebar:
    st.header("🔍 Match Setup")
    match_type = st.selectbox("Match Type", sorted(df["match_type"].dropna().astype(str).unique()))
    df_filtered = df[df["match_type"] == match_type].copy()

    # Inject MLC venue manually if not present
    if match_type == "MLC":
        mlc_venue = "Central Broward Regional Park Stadium Turf Ground"
        if mlc_venue not in df_filtered["venue"].values:
            df_filtered.loc[len(df_filtered)] = {
                "match_type": "MLC",
                "innings_1_team": "Seattle Orcas",
                "innings_2_team": "MI New York",
                "venue": mlc_venue
            }

    batting_team = st.selectbox("Batting Team", sorted(df_filtered["innings_1_team"].dropna().unique()))
    bowling_team = st.selectbox("Bowling Team", sorted(df_filtered["innings_2_team"].dropna().unique()))
    venue = st.selectbox("Venue", sorted(df_filtered["venue"].dropna().unique()))
    powerplay_runs = st.number_input("Powerplay Runs (6 overs)", min_value=0, max_value=100, step=1)
    powerplay_wkts = st.number_input("Powerplay Wickets Fallen", min_value=0, max_value=6, step=1)

# === Venue Stats ===
st.subheader(f"📊 Venue Stats: {venue}")
venue_df = df[df["venue"] == venue]

avg_score = venue_df["innings_1_runs"].mean()
avg_boundaries = venue_df["innings_1_boundaries"].mean()
boundary_std = venue_df["innings_1_boundaries"].std()

col1, col2, col3 = st.columns(3)
col1.metric("Avg 1st Inns Score", int(avg_score) if not np.isnan(avg_score) else "N/A")
col2.metric("Avg 1st Inns Boundaries", round(avg_boundaries, 1) if not np.isnan(avg_boundaries) else "N/A")
col3.metric("Boundary Std Dev", round(boundary_std, 1) if not np.isnan(boundary_std) else "N/A")

# === Line Chart ===
st.markdown("### 📈 Avg First Innings Runs per Over at Venue")
avg_runs_per_over = [venue_df.get(f"innings_1_over_{i}_runs", pd.Series([0])).mean() for i in range(1, 21)]
fig_line = px.line(x=list(range(1, 21)), y=avg_runs_per_over,
                   labels={'x': 'Over', 'y': 'Avg Runs'}, title="Avg First Innings Runs per Over")
st.plotly_chart(fig_line, use_container_width=True)

# === Pie Chart ===
st.markdown("### 🥧 Boundary Ranges at This Venue")
bucket_ranges = {"<25": 0, "25-40": 0, "41-55": 0, "55+": 0}
for val in venue_df["match_total_boundaries"].dropna():
    if val < 25:
        bucket_ranges["<25"] += 1
    elif val <= 40:
        bucket_ranges["25-40"] += 1
    elif val <= 55:
        bucket_ranges["41-55"] += 1
    else:
        bucket_ranges["55+"] += 1

pie_data = pd.DataFrame({"Bucket": list(bucket_ranges.keys()), "Matches": list(bucket_ranges.values())})
fig_pie = px.pie(pie_data, names="Bucket", values="Matches", title="Boundary Ranges")
st.plotly_chart(fig_pie, use_container_width=True)

# === Prediction ===
if st.button("🚀 Predict Now"):
    try:
        sample_row = df_filtered[
            (df_filtered["innings_1_team"] == batting_team) &
            (df_filtered["innings_2_team"] == bowling_team) &
            (df_filtered["venue"] == venue)
        ]

        if sample_row.empty:
            sample_row = df_filtered[
                (df_filtered["innings_1_team"] == batting_team) &
                (df_filtered["innings_2_team"] == bowling_team)
            ].iloc[[0]]
            sample_row["venue"] = venue
            venue_avg = df[df["venue"] == venue].mean(numeric_only=True)
            for col in ["venue_avg_first_innings_score", "venue_avg_boundaries", "venue_boundary_std"]:
                sample_row[col] = venue_avg.get(col, 0)
        else:
            sample_row = sample_row.iloc[[0]]

        venue_pp = df.groupby("venue")["innings_1_powerplay_runs"].mean().reset_index()
        venue_pp.columns = ["venue", "venue_avg_pp"]
        sample_row = sample_row.merge(venue_pp, on="venue", how="left")

        input_df = pd.DataFrame([{
            'venue_code': sample_row.get("venue_code", pd.Series([0])).values[0],
            'match_type_code': sample_row.get("match_type_code", pd.Series([0])).values[0],
            'innings_1_powerplay_runs': powerplay_runs,
            'innings_1_powerplay_wkts': powerplay_wkts,
            'batting_team_rating': sample_row.get("batting_team_rating", pd.Series([50])).values[0],
            'bowling_team_rating': sample_row.get("bowling_team_rating", pd.Series([50])).values[0],
            'venue_avg_first_innings_score': sample_row.get("venue_avg_first_innings_score", pd.Series([150])).values[0],
            'venue_avg_boundaries': sample_row.get("venue_avg_boundaries", pd.Series([15])).values[0],
            'venue_boundary_std': sample_row.get("venue_boundary_std", pd.Series([5])).values[0],
            'head_to_head_win_ratio': sample_row.get("head_to_head_win_ratio", pd.Series([0.5])).values[0],
            'last_5_avg_score': sample_row.get("last_5_avg_score", pd.Series([140])).values[0],
            'net_team_rating': sample_row.get("batting_team_rating", pd.Series([50])).values[0] - sample_row.get("bowling_team_rating", pd.Series([50])).values[0],
            'innings_1_powerplay_loss_rate': powerplay_wkts / 6,
            'net_run_rate_pp': powerplay_runs - sample_row["venue_avg_pp"].values[0],
            'collapse_flag': 1 if powerplay_wkts >= 3 else 0
        }])

        predicted_score = int(score_model.predict(input_df)[0])
        predicted_boundaries = int(boundary_model.predict(input_df)[0])
        bucket_label = bucket_label_encoder.inverse_transform(bucket_model.predict(input_df))[0]
        win_prob = float(win_model.predict_proba(win_model_scaler.transform(input_df))[0][1]) * 100

        st.success(f"🏏 Predicted 1st Innings Score: {predicted_score} runs")
        st.info(f"🔔 Predicted Boundaries: {predicted_boundaries}")
        st.warning(f"📦 Predicted Boundary Bucket: {bucket_label}")
        st.success(f"✅ Win Probability (Batting Team): {win_prob:.2f}%")

        if st.button("💾 Save Prediction"):
            prediction_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "match_type": match_type,
                "venue": venue,
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "pp_runs": powerplay_runs,
                "pp_wkts": powerplay_wkts,
                "predicted_score": predicted_score,
                "predicted_boundaries": predicted_boundaries,
                "boundary_bucket": bucket_label,
                "win_probability": win_prob
            }
            try:
                history_df = pd.read_csv("prediction_history.csv")
                history_df = pd.concat([history_df, pd.DataFrame([prediction_data])], ignore_index=True)
            except FileNotFoundError:
                history_df = pd.DataFrame([prediction_data])
            history_df.to_csv("prediction_history.csv", index=False)
            st.success("📁 Prediction saved to history!")

    except Exception as e:
        st.error(f"❌ Prediction Failed: {e}")
