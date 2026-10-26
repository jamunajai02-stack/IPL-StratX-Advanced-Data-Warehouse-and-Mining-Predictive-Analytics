
# 🏏 IPL StratX: Advanced Data Warehouse and Data Mining for Predictive Analytics

**IPL StratX** is an integrated data warehouse, data mining, and machine learning framework designed to analyze Indian Premier League (IPL) cricket data. The project combines structured data storage, player performance analytics, pattern discovery, predictive modeling, and interactive visualization to generate meaningful insights into IPL matches, teams, and players.

## 📌 Project Overview

The project aims to transform raw IPL ball-by-ball data into actionable insights through data preprocessing, data warehousing, data mining, machine learning, and interactive dashboard visualization.

It supports the analysis of player performance, team statistics, match outcomes, scoring patterns, and relationships within cricket data.

## 🎯 Objectives

- Develop an integrated data warehouse for IPL match and player data.
- Clean, preprocess, and integrate historical IPL datasets.
- Analyze player performance and identify performance-based player clusters.
- Discover interesting relationships and recurring patterns using association rule mining.
- Predict match winners and estimate final innings scores using machine learning.
- Identify important features influencing model predictions.
- Perform multidimensional analysis using OLAP techniques.
- Present analytical results through an interactive Streamlit dashboard.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Data processing, analytics, and machine learning |
| Pandas and NumPy | Data cleaning and manipulation |
| Scikit-learn | Classification, clustering, and predictive modeling |
| SQLite | Data warehouse storage and analytical queries |
| MongoDB Atlas | Cloud database for selected project datasets and mining outputs |
| Streamlit | Interactive analytics dashboard |
| Matplotlib and Seaborn | Data visualization |
| Git and GitHub | Version control and project collaboration |
| Git Large File Storage (Git LFS) | Versioning large model files |

## 📂 Project Structure

```text
IPL/
├── app.py
├── requirements.txt
├── .gitignore
├── .gitattributes
│
├── data/
│
├── mining/
│   ├── association_rules.py
│   ├── random_forest.py
│   └── score_prediction.py
│
├── models/
│   ├── data/
│   ├── match_winner_model.pkl
│   └── score_predictor_model.pkl
│
├── outputs/
│   ├── IPL_StratX_cleaned.csv
│   ├── IPL_StratX_cleaned_2008_2026.csv
│   ├── IPL_StratX_combined_2008_2026.csv
│   └── ...
│
├── preprocessing/
│   └── preprocess.py
│
├── visualizations/
│
└── warehouse/
    ├── create_warehouse.py
    ├── create_player_performance.py
    ├── olap_analysis.py
    └── verify_warehouse.py
```

*Note: This is a simplified representation of the project structure. Some directories contain additional scripts, datasets, and output files.*

## 📊 Dataset

The project integrates historical IPL data, including ball-by-ball deliveries, match details, team information, player statistics, and performance-related attributes.

The combined dataset developed during the project covers the **2008–2026 seasons**.

Previously recorded dataset statistics:

- **Matches:** 1,243
- **Ball-by-ball records:** Approximately 295,714
- **Features:** 35 columns

These figures describe the previously prepared combined dataset and may change as the data is updated.

## 🏗️ Data Warehouse

The data warehouse organizes IPL information into dimension and fact tables to support structured querying and multidimensional analysis.

### Dimension tables

- `dim_date` — Date-related information.
- `dim_team` — Team details.
- `dim_player` — Player information.
- `dim_venue` — Match venue details.
- `dim_match` — Match-related information.

### Fact tables

- `fact_ball_delivery` — Ball-by-ball match records.
- `fact_player_performance` — Player performance metrics.

This structure supports analysis across players, teams, venues, dates, matches, and delivery-level statistics.

## 🧠 Data Mining and Machine Learning

### 1. Player Performance Clustering

Uses K-Means clustering to group players according to performance-related features such as runs, strike rate, boundaries, wickets, economy, and matches played.

### 2. Match-Winner Prediction

Uses a Random Forest classification model to learn from historical match-related data and predict match outcomes based on the features provided to the model.

### 3. Innings Score Prediction

Uses a trained machine learning model to estimate the final innings score from match-state information such as runs scored, wickets lost, and legal balls bowled.

### 4. Association Rule Mining

Identifies frequent combinations and relationships in the selected IPL data to uncover potentially useful patterns.

### 5. Feature Importance Analysis

Examines the relative contribution of model features to predictions, helping users understand which inputs influence model behavior.

### 6. OLAP Analytics

Supports multidimensional analysis of IPL data across relevant dimensions, enabling comparisons of teams, players, venues, and match performance.

## 📈 Interactive Dashboard

The Streamlit application brings the project's analytical modules together in a single interface.

The dashboard includes:

- **Overview:** Summary statistics and key IPL metrics.
- **Player Analytics:** Player performance exploration and comparisons.
- **ML Insights:** Predictive modeling results and feature importance.
- **Data Mining:** Association rules and pattern-discovery outputs.
- **OLAP Analytics:** Multidimensional analysis of warehouse data.
- **Live Predictors:** Interactive prediction features using the available model inputs.

*Prediction outputs are estimates generated by trained models and should not be interpreted as guaranteed match results.*

## ⚙️ Installation and Setup

### Prerequisites

- Python installed on your system.
- Git installed.
- A MongoDB Atlas account if you intend to use the cloud database features.
- Git LFS installed to retrieve large files tracked using Git LFS.

### Step 1: Clone the repository

```bash
git clone https://github.com/jamunajai02-stack/IPL-StratX-Advanced-Data-Warehouse-and-Mining-Predictive-Analytics.git
cd IPL-StratX-Advanced-Data-Warehouse-and-Mining-Predictive-Analytics
```

### Step 2: Download Git LFS files

```bash
git lfs install
git lfs pull
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows Command Prompt:

```cmd
venv\Scripts\activate
```

Or on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Configure environment variables

If your dashboard uses MongoDB Atlas, configure the required connection settings using environment variables.

For example, in PowerShell:

```powershell
$env:MONGODB_PASSWORD="your_actual_password"
```

Replace the placeholder with your actual database password. Set any other connection variables required by your application.

**Security:** Never commit actual passwords, database credentials, API keys, or private connection strings to GitHub. Use environment variables and keep `.env` files out of version control.

### Step 6: Run the dashboard

```bash
streamlit run app.py
```

Streamlit will display the local address for opening the dashboard in your browser.

*Note: Some features may require the relevant datasets, trained models, a configured MongoDB Atlas connection, or a locally initialized warehouse database.*

## 📉 Previously Recorded Model Results

The following are previously recorded experimental results from the project:

| Evaluation metric | Recorded result |
|---|---:|
| Match-winner classification test accuracy | 65.36% |
| Score prediction MAE | 18.56 |
| Score prediction RMSE | 25.37 |
| Score prediction R² | 0.3452 |

These results depend on the dataset, features, preprocessing, train-test split, and model configuration used in the recorded experiments. They are not a guarantee of future performance.

## 🔮 Future Enhancements

- Integrate regularly updated IPL datasets.
- Improve prediction performance through feature engineering and model comparison.
- Add richer player comparison and team-performance visualizations.
- Expand interactive filtering and multidimensional OLAP reports.
- Improve deployment and database configuration.
- Add automated data validation and model evaluation workflows.
- Explore real-time data integration where reliable data sources are available.

## 👩‍💻 Contributors

Developed as an academic project focused on data warehousing, data mining, predictive analytics, and sports data visualization.

Add the names of the project team members here.

## 📜 License

No license has been specified yet. If you intend to allow others to reuse or modify this project, consider adding an appropriate open-source license.

---

**IPL StratX — Transforming IPL Data into Analytical Insights.**
