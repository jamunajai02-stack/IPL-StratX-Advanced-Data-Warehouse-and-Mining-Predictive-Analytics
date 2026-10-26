
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

# Load the match-level experiment data source
DATA_PATH = r"outputs\IPL_StratX_cleaned_2008_2026.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

for col in ["innings", "total_runs", "is_wicket",
            "is_boundary", "extras", "legal_delivery"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if "is_super_over" in df.columns:
    df = df[df["is_super_over"].fillna(0) == 0]

records = []

for match_id, g in df.groupby("match_id", sort=False):
    first = g[g["innings"] == 1]
    second = g[g["innings"] == 2]

    if first.empty or second.empty:
        continue

    team1 = first["batting_team"].iloc[0]
    team2 = second["batting_team"].iloc[0]

    if pd.isna(team1) or pd.isna(team2) or team1 == team2:
        continue

    runs = first["total_runs"].sum()
    second_runs = second["total_runs"].sum()
    legal_balls = first["legal_delivery"].fillna(0).sum()

    records.append({
        "match_id": match_id,
        "season": first["season"].iloc[0],
        "first_runs": runs,
        "first_wickets": first["is_wicket"].fillna(0).sum(),
        "first_extras": first["extras"].fillna(0).sum(),
        "first_boundaries": first["is_boundary"].fillna(0).sum(),
        "first_run_rate": runs / legal_balls * 6 if legal_balls else 0,
        "legal_balls": legal_balls,
        "target": int(runs > second_runs),
    })

matches = pd.DataFrame(records)
matches["season"] = pd.to_numeric(matches["season"], errors="coerce")
matches = matches.dropna(subset=["season"])

features = [
    "first_runs",
    "first_wickets",
    "first_extras",
    "first_boundaries",
    "first_run_rate",
    "legal_balls",
]

# Use a chronological holdout: the last three seasons.
seasons = sorted(matches["season"].unique())
test_seasons = seasons[-3:]

train = matches[~matches["season"].isin(test_seasons)]
test = matches[matches["season"].isin(test_seasons)]

X_train, y_train = train[features], train["target"]
X_test, y_test = test[features], test["target"]

print("Training matches:", len(train))
print("Testing matches:", len(test))
print("Test seasons:", test_seasons)

# Compare several settings; keep the same test set.
models = {
    "Baseline": DummyClassifier(strategy="most_frequent"),

    "Random Forest 300": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    ),

    "Random Forest 600": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=600,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    ),

    "Random Forest Deeper": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_leaf=2,
            class_weight=None,
            random_state=21,
            n_jobs=-1,
        ),
    ),

    "Extra Trees": make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    ),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    results.append({
        "Model": name,
        "Accuracy (%)": round(acc * 100, 2),
    })

    print(f"\n{name}: {acc * 100:.2f}%")
    print(classification_report(y_test, pred, zero_division=0))

result_df = pd.DataFrame(results).sort_values(
    "Accuracy (%)", ascending=False
)
print("\n===== RESULTS =====")
print(result_df.to_string(index=False))

result_df.to_csv(
    r"outputs\winner_tuning_results.csv",
    index=False,
)
print("\nSaved outputs\\winner_tuning_results.csv")
