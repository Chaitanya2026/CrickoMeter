import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# === Load models from current directory (no 'models/' folder) ===
score_model = joblib.load("models/score_model.pkl")
boundary_model = joblib.load("models/boundary_model.pkl")
bucket_model = joblib.load("models/bucket_model.pkl")
bucket_label_encoder = joblib.load("models/bucket_label_encoder.pkl")
win_model = joblib.load("models/win_model.pkl")
win_model_scaler = joblib.load("models/win_model_scaler.pkl")

# === Load data ===
df = pd.read_csv("combined_t20i_and_ipl_matches.csv")

# === Fix venue names ===
venue_map = {
    "Dr. DY Patil Sports Academy": "Dr. DY Patil Sports Academy, Mumbai",
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium, Bengaluru",
    "MA Chidambaram Stadium , Chepauk": "MA Chidambaram Stadium, Chepauk, Chennai",
    "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh",
    "Punjab Crciket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh"
}
df['venue'] = df['venue'].replace(venue_map)

# === Drop legacy teams ===
old_teams = ['Pune Warriors', 'Deccan Chargers', 'Gujarat Lions',
             'Rising Pune Supergiant', 'Rising Pune Supergiants', 'Kochi Tuskers Kerala']
df = df[~df['innings_1_team'].isin(old_teams)]
df = df[~df['innings_2_team'].isin(old_teams)]

# === Streamlit UI ===
st.set_page_config(page_title="🏏 Cricket Predictor", layout="wide")
st.title("🏏 Cricket Match Predictor")

with st.sidebar:
    st.header("🔍 Match Setup")
    match_type = st.selectbox("Match Type", sorted(df["match_type"].dropna().unique()))
    filtered_df = df[df["match_type"] == match_type]

    batting_team = st.selectbox("Batting Team", sorted(filtered_df["innings_1_team"].dropna().unique()))
    bowling_team = st.selectbox("Bowling Team", sorted(filtered_df["innings_2_team"].dropna().unique()))
    venue = st.selectbox("Venue", sorted(filtered_df["venue"].dropna().unique()))

    powerplay_runs = st.number_input("Powerplay Runs (6 overs)", min_value=0, max_value=100, step=1)
    powerplay_wkts = st.number_input("Powerplay Wickets Fallen", min_value=0, max_value=6, step=1)

# === Venue Stats ===
st.subheader(f"📊 Venue Stats: {venue}")
venue_df = df[df["venue"] == venue]

col1, col2, col3 = st.columns(3)
col1.metric("Avg 1st Inns Score", int(venue_df["innings_1_runs"].mean()))
col2.metric("Avg. 1st Inns Boundaries", round(venue_df["innings_1_boundaries"].mean(), 1))
col3.metric("Boundary Std Dev", round(venue_df["innings_1_boundaries"].std(), 1))

# === Line Chart ===
st.markdown("### 📈 Avg First Innings Runs per Over at Venue")
avg_runs_per_over = []
for over in range(1, 21):
    col_name = f"innings_1_over_{over}_runs"
    avg_runs_per_over.append(venue_df[col_name].mean() if col_name in venue_df.columns else 0)

fig_line = px.line(
    x=list(range(1, 21)),
    y=avg_runs_per_over,
    labels={'x': 'Over', 'y': 'Avg Runs'},
    title="Avg First Innings Runs per Over"
)
st.plotly_chart(fig_line, use_container_width=True)

# === Pie Chart ===
st.markdown("### 🥧 Boundary Ranges at This Venue")
bucket_ranges = {"<25": 0, "25-40": 0, "41-55": 0, "55+": 0}
for val in venue_df["match_total_boundaries"]:
    if val < 25:
        bucket_ranges["<25"] += 1
    elif val <= 40:
        bucket_ranges["25-40"] += 1
    elif val <= 55:
        bucket_ranges["41-55"] += 1
    else:
        bucket_ranges["55+"] += 1

pie_data = pd.DataFrame({
    "Bucket": list(bucket_ranges.keys()),
    "Matches": list(bucket_ranges.values())
})
fig_pie = px.pie(pie_data, names="Bucket", values="Matches", title="Boundary Ranges")
st.plotly_chart(fig_pie, use_container_width=True)

# === Prediction Logic ===
if st.button("🚀 Predict Now"):
    try:
        sample_row = df[
            (df["match_type"] == match_type) &
            (df["innings_1_team"] == batting_team) &
            (df["innings_2_team"] == bowling_team) &
            (df["venue"] == venue)
        ]

        if sample_row.empty:
            sample_row = df[
                (df["innings_1_team"] == batting_team) &
                (df["innings_2_team"] == bowling_team)
            ].iloc[[0]]
            sample_row["venue"] = venue
            venue_avg = df[df["venue"] == venue].mean(numeric_only=True)
            for col in ["venue_avg_first_innings_score", "venue_avg_boundaries", "venue_boundary_std"]:
                sample_row[col] = venue_avg[col]
        else:
            sample_row = sample_row.iloc[[0]]

        # Add venue_avg_pp
        venue_pp = df.groupby("venue")["innings_1_powerplay_runs"].mean().reset_index()
        venue_pp.columns = ["venue", "venue_avg_pp"]
        sample_row = sample_row.merge(venue_pp, on="venue", how="left")

        input_df = pd.DataFrame([{
            'venue_code': sample_row["venue_code"].values[0],
            'match_type_code': sample_row["match_type_code"].values[0],
            'innings_1_powerplay_runs': powerplay_runs,
            'innings_1_powerplay_wkts': powerplay_wkts,
            'batting_team_rating': sample_row["batting_team_rating"].values[0],
            'bowling_team_rating': sample_row["bowling_team_rating"].values[0],
            'venue_avg_first_innings_score': sample_row["venue_avg_first_innings_score"].values[0],
            'venue_avg_boundaries': sample_row["venue_avg_boundaries"].values[0],
            'venue_boundary_std': sample_row["venue_boundary_std"].values[0],
            'head_to_head_win_ratio': sample_row["head_to_head_win_ratio"].values[0],
            'last_5_avg_score': sample_row["last_5_avg_score"].values[0],
            'net_team_rating': sample_row["batting_team_rating"].values[0] - sample_row["bowling_team_rating"].values[0],
            'innings_1_powerplay_loss_rate': powerplay_wkts / 6,
            'net_run_rate_pp': powerplay_runs - sample_row["venue_avg_pp"].values[0],
            'collapse_flag': 1 if powerplay_wkts >= 3 else 0
        }])

        # Feature validation
        expected_features = ['venue_code', 'match_type_code', 'innings_1_powerplay_runs', 'innings_1_powerplay_wkts',
                             'batting_team_rating', 'bowling_team_rating', 'venue_avg_first_innings_score',
                             'venue_avg_boundaries', 'venue_boundary_std', 'head_to_head_win_ratio',
                             'last_5_avg_score', 'net_team_rating', 'innings_1_powerplay_loss_rate',
                             'net_run_rate_pp', 'collapse_flag']
        actual_features = list(input_df.columns)

        if set(expected_features) != set(actual_features):
            st.error(f"❌ Feature mismatch!\nExpected: {expected_features}\nGot: {actual_features}")
        else:
            # Run predictions
            predicted_score = int(score_model.predict(input_df)[0])
            predicted_boundaries = int(boundary_model.predict(input_df)[0])
            bucket_label = bucket_label_encoder.inverse_transform(bucket_model.predict(input_df))[0]

            win_features_scaled = win_model_scaler.transform(input_df)
            win_prob = float(win_model.predict_proba(win_features_scaled)[0][1]) * 100

            st.success(f"🏏 Predicted 1st Innings Score: {predicted_score} runs")
            st.info(f"🔔 Predicted Boundaries: {predicted_boundaries}")
            st.warning(f"📦 Predicted Boundary Bucket: {bucket_label}")
            st.success(f"✅ Win Probability (Batting Team): {win_prob:.2f}%")

    except Exception as e:
        st.error(f"❌ Prediction Failed: {e}")
