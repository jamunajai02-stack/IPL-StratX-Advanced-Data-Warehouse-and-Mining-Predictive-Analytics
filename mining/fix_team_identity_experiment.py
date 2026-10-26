
import pandas as pd

INPUT_FILE = r"outputs/IPL_StratX_cleaned_2008_2026.csv"
OUTPUT_FILE = r"outputs/IPL_StratX_team_identity_experiment.csv"

print("Loading IPL dataset...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

# Correct the historical franchise identity.
# The dataset uses the SRH name for Deccan Chargers in 2008–2012.
old_srh_mask = (
    (pd.to_numeric(df["season"], errors="coerce") < 2013)
    & (df["batting_team"] == "Sunrisers Hyderabad")
)
old_srh_bowling_mask = (
    (pd.to_numeric(df["season"], errors="coerce") < 2013)
    & (df["bowling_team"] == "Sunrisers Hyderabad")
)

df.loc[old_srh_mask, "batting_team"] = "Deccan Chargers"
df.loc[old_srh_bowling_mask, "bowling_team"] = "Deccan Chargers"

# Treat the Bangalore and Bengaluru spellings as one franchise.
df["batting_team"] = df["batting_team"].replace(
    "Royal Challengers Bengaluru", "Royal Challengers Bangalore"
)
df["bowling_team"] = df["bowling_team"].replace(
    "Royal Challengers Bengaluru", "Royal Challengers Bangalore"
)

df.to_csv(OUTPUT_FILE, index=False)

print("\nCorrected dataset saved to:", OUTPUT_FILE)
print("\nHistorical franchise verification:")
for season in [2008, 2012, 2013, 2016, 2022, 2026]:
    season_data = df[pd.to_numeric(df["season"], errors="coerce") == season]
    teams = sorted(
        set(season_data["batting_team"].dropna())
        | set(season_data["bowling_team"].dropna())
    )
    print(f"{season}: {', '.join(teams)}")

print("\nOriginal dataset was not modified.")
