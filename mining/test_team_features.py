
import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

INPUT_FILE = r"outputs/IPL_StratX_team_identity_experiment.csv"
OUTPUT_DIR = r"outputs"

print("Loading experimental dataset...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

required = [
    "season", "match_id", "innings", "batting_team",
    "bowling_team", "total_runs"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Convert numeric columns safely.
for col in [
    "season", "innings", "total_runs", "runs_off_bat",
    "extras", "is_wicket", "is_boundary", "legal_delivery"
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Keep the two regular innings only; super overs are not used.
df = df[df["innings"].isin([1, 2])].copy()

# Build one row per match innings.
group_cols = ["season", "match_id", "innings"]

agg = {
    "batting_team": "first",
    "bowling_team": "first",
    "total_runs": "sum",
}
if "venue" in df.columns:
    agg["venue"] = "first"

for col in [
    "runs_off_bat", "extras", "is_wicket",
    "is_boundary", "legal_delivery"
]:
    if col in df.columns:
        agg[col] = "sum"

innings_data = (
    df.groupby(group_cols, dropna=False)
      .agg(agg)
      .reset_index()
)

# Keep only matches with exactly one usable row for each innings.
first = innings_data[innings_data["innings"] == 1].copy()
second = innings_data[innings_data["innings"] == 2].copy()

first = first.rename(columns={
    c: f"first_{c}"
    for c in first.columns
    if c not in ["season", "match_id", "innings"]
})
second = second.rename(columns={
    c: f"second_{c}"
    for c in second.columns
    if c not in ["season", "match_id", "innings"]
})

matches = first.merge(
    second,
    on=["season", "match_id"],
    how="inner",
    validate="one_to_one"
)

# Exclude tied scores because this simple target does not model super overs.
matches = matches[
    matches["first_total_runs"] != matches["second_total_runs"]
].copy()

matches["first_team_wins"] = (
    matches["first_total_runs"] > matches["second_total_runs"]
).astype(int)

# Build first-innings features.
matches["first_runs"] = matches["first_total_runs"]

for col in ["is_wicket", "extras", "is_boundary",
            "legal_delivery", "runs_off_bat"]:
    source = f"first_{col}"
    if source not in matches.columns:
        matches[source] = 0
    matches[source] = pd.to_numeric(
        matches[source], errors="coerce"
    ).fillna(0)

matches["first_wickets"] = matches["first_is_wicket"]
matches["first_extras"] = matches["first_extras"]
matches["first_boundaries"] = matches["first_is_boundary"]
matches["legal_balls"] = matches["first_legal_delivery"]
matches["bat_runs"] = matches["first_runs_off_bat"]

matches["first_run_rate"] = (
    matches["first_runs"] /
    (matches["legal_balls"].replace(0, pd.NA) / 6)
).fillna(0)

numeric_features = [
    "first_runs",
    "first_wickets",
    "first_extras",
    "first_boundaries",
    "first_run_rate",
    "legal_balls",
    "bat_runs",
]

categorical_features = [
    "first_batting_team",
    "second_batting_team",
]

if "first_venue" in matches.columns:
    categorical_features.append("first_venue")

# Avoid rows with missing team identities.
matches = matches.dropna(
    subset=["first_batting_team", "second_batting_team"]
).copy()

# Hold out complete recent seasons.
train = matches[matches["season"] <= 2023].copy()
test = matches[matches["season"].isin([2024, 2025, 2026])].copy()

if train.empty or test.empty:
    raise ValueError(
        f"Empty split: train={len(train)}, test={len(test)}. "
        "Check season values in the dataset."
    )

X_train_num = train[numeric_features].fillna(0)
X_test_num = test[numeric_features].fillna(0)
y_train = train["first_team_wins"]
y_test = test["first_team_wins"]

# Model A: first-innings statistics only.
model_a = RandomForestClassifier(
    n_estimators=600,
    random_state=42,
    min_samples_leaf=2,
    n_jobs=-1
)
model_a.fit(X_train_num, y_train)
pred_a = model_a.predict(X_test_num)

# Model B: statistics + team identity + venue.
all_features = numeric_features + categorical_features

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
    ]
)

model_b = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    ))
])

model_b.fit(train[all_features], y_train)
pred_b = model_b.predict(test[all_features])

# Majority-class baseline.
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train_num, y_train)
pred_base = baseline.predict(X_test_num)

results = pd.DataFrame([
    {
        "model": "Majority baseline",
        "accuracy_percent": accuracy_score(y_test, pred_base) * 100
    },
    {
        "model": "First-innings statistics",
        "accuracy_percent": accuracy_score(y_test, pred_a) * 100
    },
    {
        "model": "Statistics + teams + venue",
        "accuracy_percent": accuracy_score(y_test, pred_b) * 100
    },
])

results["accuracy_percent"] = results["accuracy_percent"].round(2)
results.to_csv(
    os.path.join(OUTPUT_DIR, "winner_team_feature_results.csv"),
    index=False
)

predictions = test[["season", "match_id"]].copy()
predictions["actual_first_team_wins"] = y_test.values
predictions["prediction_model_a"] = pred_a
predictions["prediction_model_b"] = pred_b
predictions.to_csv(
    os.path.join(OUTPUT_DIR, "winner_team_feature_predictions.csv"),
    index=False
)

print("\n========== DATA CHECK ==========")
print("Training matches (seasons 2008-2023):", len(train))
print("Test matches (seasons 2024-2026):", len(test))
print("Training seasons:", sorted(train["season"].unique()))
print("Test seasons:", sorted(test["season"].unique()))

print("\n========== ACCURACY COMPARISON ==========")
print(results.to_string(index=False))

print("\n========== MODEL B REPORT ==========")
print(classification_report(
    y_test,
    pred_b,
    target_names=["Second-innings team wins", "First-innings team wins"],
    zero_division=0
))

print("\nSaved:")
print("outputs/winner_team_feature_results.csv")
print("outputs/winner_team_feature_predictions.csv")
print("\nExisting dashboard and model files were not modified.")
