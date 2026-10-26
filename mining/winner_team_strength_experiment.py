
from pathlib import Path

import pandas as pd

DATA_PATH = Path("outputs/IPL_StratX_cleaned_2008_2026.csv")

df = pd.read_csv(DATA_PATH, low_memory=False)

required = [
    "match_id",
    "season",
    "innings",
    "batting_team",
    "bowling_team",
    "total_runs",
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required).copy()
df["innings"] = pd.to_numeric(df["innings"], errors="coerce")
df["total_runs"] = pd.to_numeric(df["total_runs"], errors="coerce")
df = df.dropna(subset=["innings", "total_runs"])
df["innings"] = df["innings"].astype(int)

# Aggregate each team's innings score in each match.
innings_scores = (
    df[df["innings"].isin([1, 2])]
    .groupby(
        ["season", "match_id", "innings", "batting_team", "bowling_team"],
        as_index=False,
    )
    .agg(innings_runs=("total_runs", "sum"))
)

# A standard match should have one innings-1 and one innings-2 team.
first = innings_scores[innings_scores["innings"] == 1].copy()
second = innings_scores[innings_scores["innings"] == 2].copy()

first = first.rename(
    columns={
        "batting_team": "first_team",
        "bowling_team": "second_team",
        "innings_runs": "first_runs",
    }
)

second = second.rename(
    columns={
        "batting_team": "second_team_actual",
        "bowling_team": "first_team_actual",
        "innings_runs": "second_runs",
    }
)

matches = first[
    ["season", "match_id", "first_team", "second_team", "first_runs"]
].merge(
    second[
        ["season", "match_id", "second_team_actual",
         "first_team_actual", "second_runs"]
    ],
    on=["season", "match_id"],
    how="inner",
)

matches = matches[
    (matches["first_team"] == matches["first_team_actual"])
    & (matches["second_team"] == matches["second_team_actual"])
].copy()

matches["first_team_wins"] = (
    matches["first_runs"] > matches["second_runs"]
).astype(int)

matches = matches.sort_values(["season", "match_id"]).reset_index(drop=True)

print("Valid two-innings matches:", len(matches))
print("Seasons:", sorted(matches["season"].unique()))
print("\nMatches per season:")
print(matches.groupby("season")["match_id"].nunique().to_string())

# Calculate each team's previous-match record.
# shift(1) ensures the current match result is never used as a feature.
team_rows = pd.concat(
    [
        matches[["season", "match_id", "first_team", "first_runs",
                 "second_runs", "first_team_wins"]].rename(
            columns={
                "first_team": "team",
                "first_runs": "runs_scored",
                "second_runs": "runs_conceded",
                "first_team_wins": "won",
            }
        ),
        matches[["season", "match_id", "second_team", "second_runs",
                 "first_runs", "first_team_wins"]].rename(
            columns={
                "second_team": "team",
                "second_runs": "runs_scored",
                "first_runs": "runs_conceded",
                "first_team_wins": "first_team_wins_temp",
            }
        ).assign(
            won=lambda x: 1 - x["first_team_wins_temp"]
        ).drop(columns=["first_team_wins_temp"]),
    ],
    ignore_index=True,
)

team_rows = team_rows.sort_values(["season", "match_id"])
team_rows["previous_matches"] = (
    team_rows.groupby("team").cumcount()
)
team_rows["prior_win_rate"] = (
    team_rows.groupby("team")["won"]
    .transform(lambda s: s.shift(1).expanding().mean())
)
team_rows["prior_runs_scored"] = (
    team_rows.groupby("team")["runs_scored"]
    .transform(lambda s: s.shift(1).expanding().mean())
)
team_rows["prior_runs_conceded"] = (
    team_rows.groupby("team")["runs_conceded"]
    .transform(lambda s: s.shift(1).expanding().mean())
)

print("\nTeam-history feature preview:")
print(
    team_rows[
        ["season", "match_id", "team", "previous_matches",
         "prior_win_rate", "prior_runs_scored", "prior_runs_conceded"]
    ].head(15).to_string(index=False)
)

print("\nNo existing model or dashboard files were changed.")
