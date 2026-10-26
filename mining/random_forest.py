
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. CREATE OUTPUT FOLDERS
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --------------------------------------------------
# 2. LOAD CLEANED DATA
# --------------------------------------------------

df = pd.read_csv(
    "outputs/IPL_StratX_cleaned_2008_2026.csv",
    low_memory=False
)

df["match_id"] = pd.to_numeric(
    df["match_id"], errors="coerce"
)
df["innings"] = pd.to_numeric(
    df["innings"], errors="coerce"
)

df = df.dropna(subset=["match_id", "innings"]).copy()

df["match_id"] = df["match_id"].astype(int)
df["innings"] = df["innings"].astype(int)

print("Seasons included:", sorted(df["season"].dropna().unique()))
print("Ball-by-ball records loaded:", len(df))

# --------------------------------------------------
# 3. PREPARE NUMERIC COLUMNS
# --------------------------------------------------

for column in ["total_runs", "runs_off_bat", "extras"]:
    df[column] = pd.to_numeric(
        df[column], errors="coerce"
    ).fillna(0)


def to_binary(series):
    return (
        series.astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
        .astype(int)
    )


for column in ["is_wicket", "is_boundary", "legal_delivery"]:
    df[column] = to_binary(df[column])

# --------------------------------------------------
# 4. AGGREGATE INNINGS STATISTICS
# --------------------------------------------------

innings_stats = df.groupby(["match_id", "innings"]).agg(
    team=("batting_team", "first"),
    runs=("total_runs", "sum"),
    wickets=("is_wicket", "sum"),
    boundaries=("is_boundary", "sum"),
    extras=("extras", "sum"),
    legal_balls=("legal_delivery", "sum")
).reset_index()

innings_stats["run_rate"] = (
    innings_stats["runs"]
    / (innings_stats["legal_balls"] / 6).replace(0, 1)
)

first = innings_stats[innings_stats["innings"] == 1].copy()
second = innings_stats[innings_stats["innings"] == 2].copy()

first = first.rename(columns={
    "team": "first_team",
    "runs": "first_runs",
    "wickets": "first_wickets",
    "boundaries": "first_boundaries",
    "extras": "first_extras",
    "legal_balls": "first_legal_balls",
    "run_rate": "first_run_rate"
})

second = second.rename(columns={
    "team": "second_team",
    "runs": "second_runs"
})

data = first.merge(
    second[["match_id", "second_team", "second_runs"]],
    on="match_id",
    how="inner"
)

# Exclude same-team records and tied scores.
data = data[
    (data["first_team"] != data["second_team"])
    & (data["first_runs"] != data["second_runs"])
].copy()

# 1 = team batting first wins; 0 = team batting first loses.
data["first_team_wins"] = (
    data["first_runs"] > data["second_runs"]
).astype(int)

features = [
    "first_runs",
    "first_wickets",
    "first_boundaries",
    "first_extras",
    "first_run_rate"
]

X = data[features]
y = data["first_team_wins"]

print("Matches available:", len(data))
print("\nTarget distribution:")
print(y.value_counts())

if len(data) < 10 or y.nunique() < 2:
    raise ValueError(
        "Not enough valid matches or target classes to train."
    )

# --------------------------------------------------
# 5. TRAIN AND EVALUATE
# --------------------------------------------------

stratify_target = y if y.value_counts().min() >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=stratify_target
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nRandom Forest Classification Results")
print("Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["First team loses", "First team wins"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# --------------------------------------------------
# 6. SAVE MODEL AND FEATURE ORDER
# --------------------------------------------------

model_path = "models/match_winner_model.pkl"

joblib.dump(
    {
        "model": model,
        "features": features,
        "target_description": {
            "0": "First innings team loses",
            "1": "First innings team wins"
        },
        "accuracy": float(accuracy)
    },
    model_path
)

print(f"\nSaved trained model: {model_path}")

# --------------------------------------------------
# 7. SAVE FEATURE IMPORTANCE AND TRAINING DATA
# --------------------------------------------------

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_csv(
    "outputs/IPL_StratX_feature_importance.csv",
    index=False
)

data.to_csv(
    "outputs/IPL_StratX_classification_data.csv",
    index=False
)

print("\nFeature Importance:")
print(importance.to_string(index=False))

print("\nSaved classification dataset and feature importance.")
print("WINNER MODEL TRAINING COMPLETE")
