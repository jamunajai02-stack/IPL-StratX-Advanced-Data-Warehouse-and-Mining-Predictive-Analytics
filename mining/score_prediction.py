
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create folders if they do not exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load cleaned IPL data
df = pd.read_csv(
    "outputs/IPL_StratX_cleaned_2008_2026.csv",
    low_memory=False
)

# Prepare match and innings identifiers
for col in ["match_id", "innings"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["match_id", "innings"]).copy()

df["match_id"] = df["match_id"].astype(int)
df["innings"] = df["innings"].astype(int)

print("Seasons included:", sorted(df["season"].dropna().unique()))
print("Ball-by-ball records loaded:", len(df))

# Convert required numeric columns
for col in ["ball", "total_runs", "extras", "runs_off_bat"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)


# Convert boolean-like values to 0 or 1
def to_flag(series):
    return series.astype(str).str.lower().isin(
        ["true", "1", "yes"]
    ).astype(int)


df["is_wicket"] = to_flag(df["is_wicket"])
df["is_boundary"] = to_flag(df["is_boundary"])
df["legal_delivery"] = to_flag(df["legal_delivery"])

# Use the first two innings only
df = df[df["innings"].isin([1, 2])].copy()

# Sort deliveries within each match and innings
df = df.sort_values(
    ["match_id", "innings", "ball"]
).copy()

groups = ["match_id", "innings"]

# Calculate cumulative innings statistics
df["runs_so_far"] = df.groupby(groups)["total_runs"].cumsum()
df["wickets_so_far"] = df.groupby(groups)["is_wicket"].cumsum()
df["boundaries_so_far"] = df.groupby(groups)["is_boundary"].cumsum()
df["extras_so_far"] = df.groupby(groups)["extras"].cumsum()
df["legal_balls_so_far"] = df.groupby(groups)["legal_delivery"].cumsum()

# Final innings score is the target
df["final_score"] = df.groupby(groups)["total_runs"].transform("sum")

# Keep checkpoints after every completed 6 legal deliveries
checkpoints = df[
    (df["legal_balls_so_far"] >= 6) &
    (df["legal_balls_so_far"] % 6 == 0)
].copy()

# Keep one checkpoint per match, innings, and legal-ball count
checkpoints = checkpoints.drop_duplicates(
    subset=["match_id", "innings", "legal_balls_so_far"],
    keep="last"
)

# Calculate current run rate
checkpoints["current_run_rate"] = (
    checkpoints["runs_so_far"] /
    (checkpoints["legal_balls_so_far"] / 6)
)

# Feature order used by both training and dashboard prediction
features = [
    "runs_so_far",
    "wickets_so_far",
    "boundaries_so_far",
    "extras_so_far",
    "legal_balls_so_far",
    "current_run_rate"
]

checkpoints = checkpoints.dropna(
    subset=features + ["final_score", "match_id"]
)

print("Available score checkpoints:", len(checkpoints))

if len(checkpoints) < 20 or checkpoints["match_id"].nunique() < 2:
    raise ValueError(
        "Not enough innings checkpoints to train. "
        "Check the ball and legal_delivery columns."
    )

X = checkpoints[features]
y = checkpoints["final_score"]
match_groups = checkpoints["match_id"]

# Keep checkpoints from the same match in the same split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=match_groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# Train Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    min_samples_leaf=2
)

model.fit(X_train, y_train)

# Evaluate on held-out matches
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nRandom Forest Score Prediction Results")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Save test-set predictions
results = checkpoints.iloc[test_idx][
    [
        "match_id",
        "innings",
        "legal_balls_so_far",
        "runs_so_far",
        "wickets_so_far",
        "final_score"
    ]
].copy()

results["predicted_final_score"] = predictions

results.to_csv(
    "outputs/IPL_StratX_score_predictions.csv",
    index=False
)

# Save the trained model and required feature information
model_bundle = {
    "model": model,
    "features": features,
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "training_description": (
        "Predicts final innings score from current innings statistics"
    )
}

joblib.dump(
    model_bundle,
    "models/score_predictor_model.pkl"
)

print("\nSample predictions:")
print(results.head(10).to_string(index=False))

print("\nSaved predictions: outputs/IPL_StratX_score_predictions.csv")
print("Saved model: models/score_predictor_model.pkl")
print("SCORE PREDICTION MODEL TRAINING COMPLETE")
