# 🏏 CrickoMeter

CrickoMeter is a **data-driven T20 cricket match prediction engine** that uses deep statistical insights and machine learning to forecast match outcomes, boundary buckets, total runs, and more. It combines real-world cricketing trends with cutting-edge ML to serve predictions with up to **83% accuracy on unseen matches**.

🌐 **Live Demo**: [https://crickometer.streamlit.app](https://crickometer.streamlit.app)  
📊 **Data Source**: [Cricbuzz API & Scorecards (Extracted)]  
📌 **Built Using**: Python, Pandas, XGBoost, Streamlit, Power BI

---

## 🔍 Dashboard Insights

The project features robust Power BI dashboards, offering deep contextual cricketing insights across **T20 Internationals (T20Is)** and **Indian Premier League (IPL)** matches:

---

### 📈 General Match Insights

| Metric | Value / Interpretation |
|--------|------------------------|
| Avg. 1st innings score when targets were **chased** (T20Is) | 127.31 |
| Avg. 1st innings score when targets were **defended** (T20Is) | 169.40 |
| 🔥 Year with most sixes (T20Is) | 2024 - 5540 sixes |
| 🟠 40+ dot balls in 1st innings? | Score < 200 in 98.32% cases |
| 🇮🇳 Highest Win% Batting First vs NZ | India – 88.89% |
| Avg. boundary % (T20I): **Lowest** | South Africa – 8.42% |
| Avg. 1st innings score for: | `<25` → 112.45<br>`25–40` → 152.21<br>`41–55` → 184.71<br>`55+` → 223.71 |

---

### 💡 Phase-Wise & Boundary Analysis

#### ➤ Dot Ball Distribution (Top 11 teams, T20Is)
- **Middle Overs (7–15)** show highest dot % → **41.55%**

#### ➤ IPL Highlights:
- Avg. 1st innings **powerplay** score to reach 200 → **51.80**
- Most sixes in IPL → **IPL-2025** (1,273 sixes)
- Highest avg runs by over → **18th over** (8.95)
- Lowest → **1st over** (6.20)
- 🏟️ **Best Indian venue**: Arun Jaitley Stadium – Avg 1st inns: **188.31**
- 🟢 Best 1st Innings Scoring Team → LSG (186.97)
- 🔵 Best chasing team → GT (66.67% win rate)

---

## 🧠 ML Prediction Logic

CrickoMeter uses the following machine learning models:

| Task | Model Used | Description |
|------|------------|-------------|
| Predict First Innings Score | `XGBoost Regressor` | Estimates final score using powerplay, team & venue stats |
| Predict Boundary Count | `XGBoost Regressor` | Total 4s + 6s |
| Predict Boundary Range (Bucket) | `XGBoost Classifier` | Multi-class classification: `<25`, `25-40`, etc. |
| Predict Win Probability | `Random Forest Classifier` | Binary prediction (True/False) on win based on innings stats |

📌 These models were tested on unseen real-world match data and achieved **~83% accuracy** in multi-task predictions.

---

## 📂 Notebooks Overview

### 1. `Data_Extraction.ipynb`
- Extracts match-by-match data from structured CSVs & sources
- Cleans missing values
- Engineer advanced cricketing features like:
  - Powerplay Efficiency
  - Collapse Flags
  - Team Rating Normalization
  - Dot Ball Impact & Boundary %

### 2. `Model_Training.ipynb`
- Prepares datasets for each ML task
- Scales & encodes where needed
- Trains and saves `.pkl` models using `joblib`
- Visualizes performance and error metrics

---

## 🚀 Streamlit Web App

Access the app here: 👉 [**CrickoMeter Web App**](https://crickometer.streamlit.app)

- **Live predictions** on new matches
- Input batting & bowling stats for any venue/team
- Get:
  - 📊 Expected total score
  - 🧮 Predicted boundary range
  - 🎯 Win % likelihood
- Updated model reflects real-time cricketing trends

---

## 📌 Tech Stack

- **Frontend**: Streamlit
- **Backend/ML**: Python, XGBoost, Scikit-learn
- **Data Viz**: Power BI
- **Deployment**: GitHub + Streamlit Cloud

---

## 💡 Naming the Project

The name **"CrickoMeter"** reflects a fusion of:
- **Cricket** analytics and
- A **meter**-like measure of prediction and match understanding

---

## 📬 Want to Contribute?

Feel free to fork the repo, open issues, or suggest improvements to models or insights.

---

## 📜 License

This project is open-sourced under the MIT License.

---

Made with ❤️ for cricket and data by [@Chaitanya2026](https://github.com/Chaitanya2026)
