
import os
import sqlite3
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

# -------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------

DATA_PATH = r"outputs\IPL_StratX_cleaned_2008_2026.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Run this script from D:\\IPL."
    )

df = pd.read_csv(DATA_PATH)
print("Dataset loaded:", df.shape)

# -------------------------------------------------
# 2. PREPARE MATCH-LEVEL DATA
# -------------------------------------------------

required = [
    "match_id",
    "innings",
    "batting_team",
    "total_runs",
    "is_wicket",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["innings"] = pd.to_numeric(df["innings"], errors="coerce")
df["total_runs"] = pd.to_numeric(df["total_runs"], errors="coerce")
df["is_wicket"] = pd.to_numeric(df["is_wicket"], errors="coerce").fillna(0)

first = df[df["innings"] == 1].copy()
second = df[df["innings"] == 2].copy()

first_stats = first.groupby("match_id").agg(
    first_runs=("total_runs", "sum"),
    first_wickets=("is_wicket", "sum"),
    first_team=("batting_team", "first"),
)

second_stats = second.groupby("match_id").agg(
    second_runs=("total_runs", "sum"),
    second_team=("batting_team", "first"),
)

matches = first_stats.join(second_stats, how="inner")
matches = matches.dropna(subset=["first_runs", "second_runs"])

# Keep only matches with two identifiable, different teams.
matches = matches[
    matches["first_team"].notna()
    & matches["second_team"].notna()
    & (matches["first_team"] != matches["second_team"])
].copy()

# Target: did the team batting first win?
matches["target"] = (
    matches["first_runs"] > matches["second_runs"]
).astype(int)

# Additional first-innings features.
if "runs_off_bat" in first.columns:
    first["runs_off_bat"] = pd.to_numeric(
        first["runs_off_bat"], errors="coerce"
    ).fillna(0)
    matches["bat_runs"] = first.groupby("match_id")[
        "runs_off_bat"
    ].sum()

if "extras" in first.columns:
    first["extras"] = pd.to_numeric(
        first["extras"], errors="coerce"
    ).fillna(0)
    matches["first_extras"] = first.groupby("match_id")[
        "extras"
    ].sum()

if "ball" in first.columns:
    first["ball"] = pd.to_numeric(first["ball"], errors="coerce")
    if "legal_delivery" in first.columns:
        first["legal_delivery"] = pd.to_numeric(
            first["legal_delivery"], errors="coerce"
        ).fillna(0)
        legal_balls = first.groupby("match_id")[
            "legal_delivery"
        ].sum()
    else:
        # Fallback only; may count illegal deliveries as balls.
        legal_balls = first.groupby("match_id").size()

    matches["legal_balls"] = legal_balls
    matches["first_run_rate"] = (
        matches["first_runs"] / matches["legal_balls"].replace(0, np.nan) * 6
    )

if "is_boundary" in first.columns:
    first["is_boundary"] = pd.to_numeric(
        first["is_boundary"], errors="coerce"
    ).fillna(0)
    matches["first_boundaries"] = first.groupby("match_id")[
        "is_boundary"
    ].sum()

# Optional match-level context.
for col in ["season", "venue"]:
    if col in first.columns:
        matches[col] = first.groupby("match_id")[col].first()

print("Matches prepared:", len(matches))
print("Target distribution:")
print(matches["target"].value_counts())

# -------------------------------------------------
# 3. SELECT FEATURES
# -------------------------------------------------

numeric_candidates = [
    "first_runs",
    "first_wickets",
    "first_extras",
    "first_boundaries",
    "first_run_rate",
    "legal_balls",
    "bat_runs",
]

numeric_features = [
    col for col in numeric_candidates if col in matches.columns
]

if len(numeric_features) < 3:
    raise ValueError(
        f"Not enough numeric features found: {numeric_features}"
    )

# Use chronological holdout when season is available.
# Earlier seasons train; later seasons test.
if "season" in matches.columns:
    matches["season"] = pd.to_numeric(matches["season"], errors="coerce")
    matches = matches.dropna(subset=["season"])
    seasons = sorted(matches["season"].unique())

    if len(seasons) >= 3:
        test_seasons = seasons[-3:]
        train_mask = ~matches["season"].isin(test_seasons)
        test_mask = matches["season"].isin(test_seasons)
        X_train = matches.loc[train_mask, numeric_features]
        X_test = matches.loc[test_mask, numeric_features]
        y_train = matches.loc[train_mask, "target"]
        y_test = matches.loc[test_mask, "target"]
        print("Test seasons:", test_seasons)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            matches[numeric_features],
            matches["target"],
            test_size=0.25,
            random_state=42,
            stratify=matches["target"],
        )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        matches[numeric_features],
        matches["target"],
        test_size=0.25,
        random_state=42,
        stratify=matches["target"],
    )

print("Features:", numeric_features)
print("Training matches:", len(X_train))
print("Testing matches:", len(X_test))

# -------------------------------------------------
# 4. COMPARE ALGORITHMS
# -------------------------------------------------

models = {
    "Baseline": DummyClassifier(strategy="most_frequent"),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    ),
    "Extra Trees": make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    ),
    "Gradient Boosting": make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingClassifier(random_state=42),
    ),
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    ),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    results.append({
        "Model": name,
        "Accuracy (%)": round(accuracy * 100, 2),
    })

    print(f"\n{name}: {accuracy * 100:.2f}%")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0,
    ))

results_df = pd.DataFrame(results).sort_values(
    "Accuracy (%)", ascending=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

os.makedirs("outputs", exist_ok=True)
results_df.to_csv(
    r"outputs\winner_model_comparison.csv",
    index=False,
)
print("\nSaved: outputs\\winner_model_comparison.csv")
print("Experiment complete. Existing model was not changed.")
