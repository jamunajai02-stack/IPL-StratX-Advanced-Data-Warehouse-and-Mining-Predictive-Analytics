import pandas as pd
import sqlite3

# Load dataset
df = pd.read_csv("ipl_2026_ball_by_ball.csv")

# Create database
conn = sqlite3.connect("ipl_stratx.db")


# =====================================================
# 1. DIMENSION TABLE - PLAYER
# =====================================================

dim_player = pd.DataFrame({
    "player_name": pd.concat([
        df["striker"],
        df["non_striker"],
        df["bowler"]
    ]).dropna().unique()
})

dim_player["player_key"] = range(1, len(dim_player) + 1)

dim_player = dim_player[
    ["player_key", "player_name"]
]

dim_player.to_sql(
    "dim_player",
    conn,
    if_exists="replace",
    index=False
)


# =====================================================
# 2. DIMENSION TABLE - TEAM
# =====================================================

dim_team = pd.DataFrame({
    "team_name": pd.concat([
        df["batting_team"],
        df["bowling_team"]
    ]).dropna().unique()
})

dim_team["team_key"] = range(1, len(dim_team) + 1)

dim_team = dim_team[
    ["team_key", "team_name"]
]

dim_team.to_sql(
    "dim_team",
    conn,
    if_exists="replace",
    index=False
)


# =====================================================
# 3. DIMENSION TABLE - VENUE
# =====================================================

dim_venue = pd.DataFrame({
    "venue": df["venue"].dropna().unique()
})

dim_venue["venue_key"] = range(1, len(dim_venue) + 1)

dim_venue = dim_venue[
    ["venue_key", "venue"]
]

dim_venue.to_sql(
    "dim_venue",
    conn,
    if_exists="replace",
    index=False
)


# =====================================================
# 4. DIMENSION TABLE - DATE
# =====================================================

dim_date = pd.DataFrame({
    "start_date": pd.to_datetime(
        df["start_date"]
    ).dropna().unique()
})

dim_date["date_key"] = range(1, len(dim_date) + 1)

dim_date["day"] = dim_date["start_date"].dt.day
dim_date["month"] = dim_date["start_date"].dt.month
dim_date["quarter"] = dim_date["start_date"].dt.quarter
dim_date["year"] = dim_date["start_date"].dt.year

dim_date.to_sql(
    "dim_date",
    conn,
    if_exists="replace",
    index=False
)


# =====================================================
# 5. FACT TABLE - MATCH PERFORMANCE
# =====================================================

fact = df.copy()

# Add Player Key
fact = fact.merge(
    dim_player,
    left_on="striker",
    right_on="player_name",
    how="left"
)

# Add Team Key
fact = fact.merge(
    dim_team,
    left_on="batting_team",
    right_on="team_name",
    how="left"
)

# Add Venue Key
fact = fact.merge(
    dim_venue,
    on="venue",
    how="left"
)

# Convert date
fact["start_date"] = pd.to_datetime(
    fact["start_date"]
)

# Add Date Key
fact = fact.merge(
    dim_date,
    on="start_date",
    how="left"
)


# Select fact table columns
fact_match_performance = fact[[
    "match_id",
    "player_key",
    "team_key",
    "venue_key",
    "date_key",
    "runs_off_bat",
    "extras",
    "wides",
    "noballs",
    "byes"
]]

# Rename measure
fact_match_performance = fact_match_performance.rename(
    columns={
        "runs_off_bat": "runs_scored"
    }
)


# Save fact table
fact_match_performance.to_sql(
    "fact_match_performance",
    conn,
    if_exists="replace",
    index=False
)


# =====================================================
# DISPLAY RESULTS
# =====================================================

print("\n======================================")
print("IPL STRATX - STAR SCHEMA")
print("======================================")

print("\nDimension Tables:")
print("1. dim_player")
print("2. dim_team")
print("3. dim_venue")
print("4. dim_date")

print("\nFact Table:")
print("5. fact_match_performance")

print("\nPlayer Dimension:")
print(dim_player.head())

print("\nTeam Dimension:")
print(dim_team.head())

print("\nVenue Dimension:")
print(dim_venue.head())

print("\nDate Dimension:")
print(dim_date.head())

print("\nFact Table:")
print(fact_match_performance.head())

print("\nStar Schema created successfully!")

conn.close()