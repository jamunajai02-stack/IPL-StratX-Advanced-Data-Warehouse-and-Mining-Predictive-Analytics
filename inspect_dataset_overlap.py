
import pandas as pd

old_path = "outputs/IPL_StratX_cleaned.csv"
new_path = "data/ball_by_ball_data.csv"

old = pd.read_csv(old_path, low_memory=False)
new = pd.read_csv(new_path, low_memory=False)

print("\n===== EXISTING DATASET =====")
print("Seasons:", sorted(old["season"].dropna().unique()))
print("Unique matches:", old["match_id"].nunique())

print("\n===== NEW HISTORICAL DATASET =====")
print("Season range:",
      new["season_id"].min(), "to", new["season_id"].max())
print("Seasons:", sorted(new["season_id"].dropna().unique()))
print("Unique matches:", new["match_id"].nunique())

print("\n===== 2026 CHECK =====")
print("2026 rows in old dataset:",
      (old["season"] == 2026).sum())
print("2026 rows in new dataset:",
      (new["season_id"] == 2026).sum())

print("\n===== NEW DATASET DUPLICATES =====")
keys = [
    "match_id", "innings", "over_number", "ball_number"
]
print("Duplicate delivery keys:",
      new.duplicated(subset=keys).sum())

print("\n===== NEW DATASET TYPES =====")
print(new[["batsman_type", "bowler_type"]].nunique())
print("\nInspection complete.")
