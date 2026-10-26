
import pandas as pd
from pathlib import Path

old_path = Path("outputs/IPL_StratX_cleaned.csv")
new_path = Path("data/ball_by_ball_data.csv")
output_path = Path("outputs/IPL_StratX_combined_2008_2026.csv")

# Load both datasets
old = pd.read_csv(old_path, low_memory=False)
hist = pd.read_csv(new_path, low_memory=False)

print("Loaded existing:", len(old))
print("Loaded historical:", len(hist))

# Convert historical data to the existing dataset's schema
historical = pd.DataFrame(index=hist.index)

historical["Match_Number"] = pd.NA
historical["Match"] = hist["match_id"].astype(str)
historical["match_id"] = hist["match_id"].astype(str)
historical["season"] = hist["season_id"]
historical["start_date"] = pd.NA
historical["venue"] = pd.NA
historical["innings"] = hist["innings"]

# Preserve the source's over and ball numbers in a display field.
# Example: over 0, ball 0 becomes 0.1.
historical["ball"] = (
    hist["over_number"].astype(str)
    + "."
    + (hist["ball_number"] + 1).astype(str)
)

historical["batting_team"] = hist["team_batting"]
historical["bowling_team"] = hist["team_bowling"]
historical["striker"] = hist["batter"]
historical["non_striker"] = hist["non_striker"]
historical["bowler"] = hist["bowler"]

historical["runs_off_bat"] = hist["batter_runs"]
historical["extras"] = hist["extras"]
historical["wides"] = hist["wide_ball_runs"]
historical["noballs"] = hist["no_ball_runs"]
historical["byes"] = hist["bye_runs"]
historical["legbyes"] = hist["leg_bye_runs"]
historical["penalty"] = hist["penalty_runs"]

historical["wicket_type"] = hist["wicket_kind"]
historical["player_dismissed"] = hist["player_out"]
historical["other_wicket_type"] = pd.NA
historical["other_player_dismissed"] = pd.NA
historical["team_score"] = pd.NA
historical["total_runs"] = hist["total_runs"]

historical["is_wicket"] = (
    hist["is_wicket"].fillna(False).astype(int)
)

historical["is_boundary"] = (
    hist["batter_runs"].isin([4, 6]).astype(int)
)

historical["legal_delivery"] = (
    ~hist["is_wide_ball"].fillna(False).astype(bool)
    & ~hist["is_no_ball"].fillna(False).astype(bool)
).astype(int)

# Match the original column order

# Preserve additional historical features
extra_columns = [
    "batsman_type",
    "bowler_type",
    "fielders_involved",
    "is_super_over",
    "over_number",
    "ball_number",
]

for col in extra_columns:
    historical[col] = hist[col]

    # These fields are unavailable in the 2026 source
    old[col] = pd.NA

# Keep all original columns plus the additional features
all_columns = list(old.columns)

for col in extra_columns:
    if col not in all_columns:
        all_columns.append(col)

historical = historical[all_columns]
old = old[all_columns]


# Keep the original dataset intact and append historical records
combined = pd.concat(
    [old, historical],
    ignore_index=True
)

# Save as a separate file
output_path.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(output_path, index=False)

print("\n===== COMBINED DATASET =====")
print("Total rows:", len(combined))
print("Total columns:", len(combined.columns))
print("Unique seasons:", sorted(combined["season"].dropna().unique()))
print("Unique matches:", combined["match_id"].nunique())
print("Saved to:", output_path)
print("\nRows per season:")
print(combined.groupby("season").size().to_string())

print("\n===== FINAL VALIDATION =====")

print("Total rows:", len(combined))
print("Expected rows:", 295714)

print("\nMissing values in important columns:")
print(
    combined[
        ["season", "match_id", "innings",
         "batting_team", "bowling_team",
         "striker", "bowler", "total_runs"]
    ].isnull().sum()
)

print("\nRows per source period:")
print("2008-2025:", (combined["season"] < 2026).sum())
print("2026:", (combined["season"] == 2026).sum())

print("\nWicket counts by season:")
print(combined.groupby("season")["is_wicket"].sum())

print("\nTotal runs by season:")
print(combined.groupby("season")["total_runs"].sum())

print("\nValidation complete.")
