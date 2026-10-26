
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.dummy import DummyClassifier

DATA_PATH = r"outputs\IPL_StratX_cleaned_2008_2026.csv"

# --------------------------------------------------
# 1. Load and clean the ball-by-ball dataset
# --------------------------------------------------
df = pd.read_csv(DATA_PATH, low_memory=False)

df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
for col in ["season", "innings", "total_runs", "runs_off_bat",
            "extras", "is_wicket", "is_boundary", "legal_delivery"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=[
        "match_id",
        "season",
        "innings",
        "batting_team",
        "bowling_team",
        "total_runs",
    ]
).copy()

# Use only regular innings, excluding super overs.
if "is_super_over" in df.columns:
    df = df[df["is_super_over"].fillna(0) == 0].copy()

# --------------------------------------------------
# 2. Build one record per match
# --------------------------------------------------
match_rows = []

for match_id, group in df.groupby("match_id", sort=False):
    first = group[group["innings"] == 1]
    second = group[group["innings"] == 2]

    if first.empty or second.empty:
        continue

    first_team = first["batting_team"].iloc[0]
    second_team = second["batting_team"].iloc[0]

    if pd.isna(first_team) or pd.isna(second_team):
        continue
    if first_team == second_team:
        continue

    first_runs = first["total_runs"].sum()
    second_runs = second["total_runs"].sum()

    first_legal = (
        first["legal_delivery"].fillna(0).sum()
        if "legal_delivery" in first.columns
        else len(first)
    )

    first_wickets = first["is_wicket"].fillna(0).sum()

    # This is a simplified dismissal-event count.
    # It may include dismissals that do not count against the bowler.
    record = {
        "match_id": match_id,
        "date": first["start_date"].min(),
        "season": first["season"].dropna().iloc[0]
            if first["season"].notna().any() else np.nan,
        "venue": str(first["venue"].iloc[0]),
        "first_team": str(first_team),
        "second_team": str(second_team),
        "first_runs": first_runs,
        "first_wickets": first_wickets,
        "first_extras": first["extras"].fillna(0).sum()
            if "extras" in first.columns else 0,
        "first_boundaries": first["is_boundary"].fillna(0).sum()
            if "is_boundary" in first.columns else 0,
        "first_run_rate": first_runs / first_legal * 6
            if first_legal > 0 else np.nan,
        "legal_balls": first_legal,
        "target": int(first_runs > second_runs),
    }
    match_rows.append(record)

matches = pd.DataFrame(match_rows)
matches = matches.sort_values(
    ["season", "date", "match_id"],
    na_position="first"
).reset_index(drop=True)

print("Matches prepared:", len(matches))
print("Date range:", matches["date"].min(), "to", matches["date"].max())

# --------------------------------------------------
# 3. Chronological historical features
# --------------------------------------------------
# Every historical rate is calculated using earlier
# matches only. Current match results are not included.

team_history = {}
venue_history = {}

team_win_rates = []
opponent_win_rates = []
venue_first_bat_rates = []

for row in matches.itertuples(index=False):
    team = row.first_team
    opponent = row.second_team
    venue = row.venue

    # Read history BEFORE updating with this match.
    team_record = team_history.get(team, [0, 0])
    opponent_record = team_history.get(opponent, [0, 0])
    venue_record = venue_history.get(venue, [0, 0])

    team_win_rates.append(
        team_record[0] / team_record[1]
        if team_record[1] else np.nan
    )
    opponent_win_rates.append(
        opponent_record[0] / opponent_record[1]
        if opponent_record[1] else np.nan
    )
    venue_first_bat_rates.append(
        venue_record[0] / venue_record[1]
        if venue_record[1] else np.nan
    )

    # Update team history after extracting features.
    first_won = int(row.target == 1)
    second_won = int(row.target == 0)

    team_history.setdefault(team, [0, 0])
    team_history[team][0] += first_won
    team_history[team][1] += 1

    team_history.setdefault(opponent, [0, 0])
    team_history[opponent][0] += second_won
    team_history[opponent][1] += 1

    # Historical proportion of matches where the team
    # batting first won at this venue.
    venue_history.setdefault(venue, [0, 0])
    venue_history[venue][0] += first_won
    venue_history[venue][1] += 1

matches["first_team_prior_win_rate"] = team_win_rates
matches["second_team_prior_win_rate"] = opponent_win_rates
matches["venue_prior_first_bat_win_rate"] = venue_first_bat_rates

# --------------------------------------------------
# 4. Select features and hold out recent seasons
# --------------------------------------------------
features = [
    "first_runs",
    "first_wickets",
    "first_extras",
    "first_boundaries",
    "first_run_rate",
    "legal_balls",
    "first_team_prior_win_rate",
    "second_team_prior_win_rate",
    "venue_prior_first_bat_win_rate",
]

matches["season"] = pd.to_numeric(matches["season"], errors="coerce")
matches = matches.dropna(subset=["season"]).copy()

seasons = sorted(matches["season"].unique())
if len(seasons) < 4:
    raise ValueError("Not enough seasons for a chronological test.")

test_seasons = seasons[-3:]
train = matches[~matches["season"].isin(test_seasons)].copy()
test = matches[matches["season"].isin(test_seasons)].copy()

X_train = train[features]
y_train = train["target"]
X_test = test[features]
y_test = test["target"]

print("Training seasons:", sorted(train["season"].unique()))
print("Test seasons:", test_seasons)
print("Training matches:", len(train))
print("Test matches:", len(test))

# --------------------------------------------------
# 5. Compare baseline and improved models
# --------------------------------------------------
models = {
    "Baseline": DummyClassifier(strategy="most_frequent"),
    "Random Forest - Existing Features": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    ),
    "Random Forest - Historical Features": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=600,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    ),
    "Extra Trees - Historical Features": make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(
            n_estimators=600,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    ),
}

results = []

for name, model in models.items():
    if name == "Random Forest - Existing Features":
        old_features = [
            "first_runs", "first_wickets", "first_extras",
            "first_boundaries", "first_run_rate", "legal_balls",
        ]
        model.fit(train[old_features], y_train)
        predictions = model.predict(test[old_features])
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    results.append({
        "Model": name,
        "Accuracy (%)": round(accuracy * 100, 2),
    })

    print(f"\n{name}: {accuracy * 100:.2f}%")
    print(classification_report(y_test, predictions, zero_division=0))

results_df = pd.DataFrame(results).sort_values(
    "Accuracy (%)", ascending=False
)
print("\n===== FINAL COMPARISON =====")
print(results_df.to_string(index=False))

os.makedirs("outputs", exist_ok=True)
results_df.to_csv(
    r"outputs\winner_historical_feature_comparison.csv",
    index=False,
)
print("\nSaved outputs\\winner_historical_feature_comparison.csv")
print("Existing model and dashboard were not modified.")
