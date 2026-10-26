import os
import sqlite3
import pandas as pd

INPUT_FILE = "outputs/IPL_StratX_cleaned_2008_2026.csv"
DB_FILE = "outputs/IPL_StratX_warehouse.db"

os.makedirs("outputs", exist_ok=True)

print("=" * 65)
print("IPL STRATX - DATA WAREHOUSE CREATION / ETL")
print("=" * 65)

# EXTRACT
print("\n1. EXTRACT - Reading cleaned CSV...")
df = pd.read_csv(INPUT_FILE, low_memory=False)
df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
print("Rows extracted:", len(df))

# TRANSFORM
print("\n2. TRANSFORM - Preparing dimensions and facts...")

# Stable dimension keys

def make_dim(values, key_name, value_name):
    vals = pd.Series(values.dropna().astype(str).unique()).sort_values().reset_index(drop=True)
    return pd.DataFrame({key_name: range(1, len(vals) + 1), value_name: vals})

# Date dimension
unique_dates = pd.DataFrame({"date": sorted(df["start_date"].dropna().dt.normalize().unique())})
dim_date = pd.DataFrame({
    "date_key": range(1, len(unique_dates) + 1),
    "full_date": unique_dates["date"].dt.strftime("%Y-%m-%d"),
    "year": unique_dates["date"].dt.year,
    "month": unique_dates["date"].dt.month,
    "day": unique_dates["date"].dt.day
})

# Team dimension
teams = sorted(set(df["batting_team"].dropna()) | set(df["bowling_team"].dropna()))
dim_team = pd.DataFrame({"team_key": range(1, len(teams) + 1), "team_name": teams})

# Player dimension
players = sorted(
    set(df["striker"].dropna())
    | set(df["non_striker"].dropna())
    | set(df["bowler"].dropna())
    | set(df["player_dismissed"].dropna())
    | set(df["other_player_dismissed"].dropna())
)
dim_player = pd.DataFrame({"player_key": range(1, len(players) + 1), "player_name": players})

# Venue dimension
venues = sorted(df["venue"].dropna().astype(str).unique())
dim_venue = pd.DataFrame({"venue_key": range(1, len(venues) + 1), "venue_name": venues})

# Match dimension
match_cols = ["match_id", "Match_Number", "Match", "season", "start_date", "venue"]
dim_match = df[match_cols].drop_duplicates("match_id").copy()
dim_match = dim_match.sort_values("match_id").reset_index(drop=True)
dim_match.insert(0, "match_key", range(1, len(dim_match) + 1))
dim_match["start_date"] = pd.to_datetime(dim_match["start_date"]).dt.strftime("%Y-%m-%d")

# Lookup maps
team_map = dict(zip(dim_team.team_name, dim_team.team_key))
player_map = dict(zip(dim_player.player_name, dim_player.player_key))
venue_map = dict(zip(dim_venue.venue_name, dim_venue.venue_key))
date_map = dict(zip(dim_date.full_date, dim_date.date_key))
match_map = dict(zip(dim_match.match_id, dim_match.match_key))

# Ball fact
fact = pd.DataFrame()
fact["match_key"] = df["match_id"].map(match_map)
fact["date_key"] = df["start_date"].dt.strftime("%Y-%m-%d").map(date_map)
fact["innings"] = df["innings"]
fact["ball"] = df["ball"]
fact["batting_team_key"] = df["batting_team"].map(team_map)
fact["bowling_team_key"] = df["bowling_team"].map(team_map)
fact["striker_key"] = df["striker"].map(player_map)
fact["non_striker_key"] = df["non_striker"].map(player_map)
fact["bowler_key"] = df["bowler"].map(player_map)
fact["venue_key"] = df["venue"].map(venue_map)
fact["runs_off_bat"] = df["runs_off_bat"]
fact["extras"] = df["extras"]
fact["total_runs"] = df["total_runs"]
fact["is_wicket"] = df["is_wicket"]
fact["is_boundary"] = df["is_boundary"]
fact["legal_delivery"] = df["legal_delivery"]
fact["ball_key"] = range(1, len(df) + 1)

# Player performance fact
records = []
for p in players:
    batting = df[df["striker"] == p]
    bowling = df[df["bowler"] == p]
    runs = int(batting["runs_off_bat"].sum())
    balls = int(batting["legal_delivery"].sum())
    fours = int((batting["runs_off_bat"] == 4).sum())
    sixes = int((batting["runs_off_bat"] == 6).sum())
    runs_conceded = int(bowling["total_runs"].sum())
    balls_bowled = int(bowling["legal_delivery"].sum())
    wickets = int(bowling["is_wicket"].sum())
    matches = int(pd.concat([batting["match_id"], bowling["match_id"]]).nunique())
    strike_rate = (runs / balls * 100) if balls else 0
    economy = (runs_conceded / balls_bowled * 6) if balls_bowled else 0
    records.append([player_map[p], runs, balls, fours, sixes, runs_conceded,
                    balls_bowled, wickets, strike_rate, economy, matches])

fact_player = pd.DataFrame(records, columns=[
    "player_key", "runs", "balls_faced", "fours", "sixes",
    "runs_conceded", "balls_bowled", "wickets", "strike_rate",
    "economy", "matches"
])

# LOAD
print("\n3. LOAD - Creating SQLite warehouse...")
if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()
cur.executescript("""
PRAGMA foreign_keys = ON;

CREATE TABLE dim_date (
    date_key INTEGER PRIMARY KEY,
    full_date TEXT NOT NULL,
    year INTEGER,
    month INTEGER,
    day INTEGER
);

CREATE TABLE dim_team (
    team_key INTEGER PRIMARY KEY,
    team_name TEXT NOT NULL UNIQUE
);

CREATE TABLE dim_player (
    player_key INTEGER PRIMARY KEY,
    player_name TEXT NOT NULL UNIQUE
);

CREATE TABLE dim_venue (
    venue_key INTEGER PRIMARY KEY,
    venue_name TEXT NOT NULL UNIQUE
);

CREATE TABLE dim_match (
    match_key INTEGER PRIMARY KEY,
    match_id INTEGER NOT NULL UNIQUE,
    match_number INTEGER,
    match_name TEXT,
    season INTEGER,
    start_date TEXT,
    venue TEXT
);

CREATE TABLE fact_ball_delivery (
    ball_key INTEGER PRIMARY KEY,
    match_key INTEGER,
    date_key INTEGER,
    innings INTEGER,
    ball REAL,
    batting_team_key INTEGER,
    bowling_team_key INTEGER,
    striker_key INTEGER,
    non_striker_key INTEGER,
    bowler_key INTEGER,
    venue_key INTEGER,
    runs_off_bat INTEGER,
    extras INTEGER,
    total_runs INTEGER,
    is_wicket INTEGER,
    is_boundary INTEGER,
    legal_delivery INTEGER,
    FOREIGN KEY(match_key) REFERENCES dim_match(match_key),
    FOREIGN KEY(date_key) REFERENCES dim_date(date_key),
    FOREIGN KEY(batting_team_key) REFERENCES dim_team(team_key),
    FOREIGN KEY(bowling_team_key) REFERENCES dim_team(team_key),
    FOREIGN KEY(striker_key) REFERENCES dim_player(player_key),
    FOREIGN KEY(non_striker_key) REFERENCES dim_player(player_key),
    FOREIGN KEY(bowler_key) REFERENCES dim_player(player_key),
    FOREIGN KEY(venue_key) REFERENCES dim_venue(venue_key)
);

CREATE TABLE fact_player_performance (
    player_key INTEGER PRIMARY KEY,
    runs INTEGER,
    balls_faced INTEGER,
    fours INTEGER,
    sixes INTEGER,
    runs_conceded INTEGER,
    balls_bowled INTEGER,
    wickets INTEGER,
    strike_rate REAL,
    economy REAL,
    matches INTEGER,
    FOREIGN KEY(player_key) REFERENCES dim_player(player_key)
);
""")

# Insert dimensions

dim_date.to_sql("dim_date", conn, if_exists="append", index=False)
dim_team.to_sql("dim_team", conn, if_exists="append", index=False)
dim_player.to_sql("dim_player", conn, if_exists="append", index=False)
dim_venue.to_sql("dim_venue", conn, if_exists="append", index=False)

dim_match_sql = dim_match.rename(columns={
    "Match_Number": "match_number",
    "Match": "match_name"
})
dim_match_sql.to_sql("dim_match", conn, if_exists="append", index=False)

fact.to_sql("fact_ball_delivery", conn, if_exists="append", index=False)
fact_player.to_sql("fact_player_performance", conn, if_exists="append", index=False)

conn.commit()

print("\n4. WAREHOUSE SUMMARY")
for table in ["dim_date", "dim_team", "dim_player", "dim_venue", "dim_match", "fact_ball_delivery", "fact_player_performance"]:
    count = pd.read_sql_query(f"SELECT COUNT(*) AS n FROM {table}", conn)["n"][0]
    print(f"{table:28s}: {count}")

conn.close()

print("\nWarehouse created successfully:")
print(DB_FILE)
print("=" * 65)
print("ETL + DATA WAREHOUSE COMPLETE")
print("=" * 65)