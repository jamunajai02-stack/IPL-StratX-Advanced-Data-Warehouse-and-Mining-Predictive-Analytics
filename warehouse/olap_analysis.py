
import sqlite3
import pandas as pd

DB_FILE = "outputs/IPL_StratX_warehouse.db"
conn = sqlite3.connect(DB_FILE)

print("=" * 75)
print("IPL STRATX - OLAP ANALYSIS")
print("=" * 75)

# 1. ROLL-UP: Delivery level -> Team level
print("\n1. ROLL-UP - Total Runs by Team")

rollup = pd.read_sql_query("""
SELECT
    t.team_name AS team,
    SUM(f.total_runs) AS total_runs
FROM fact_ball_delivery f
JOIN dim_team t
    ON f.batting_team_key = t.team_key
GROUP BY t.team_name
ORDER BY total_runs DESC;
""", conn)

print(rollup.to_string(index=False))


# 2. DRILL-DOWN: Team -> Player details
print("\n2. DRILL-DOWN - Player Details for Top Team")

top_team = rollup.iloc[0]["team"]

drilldown = pd.read_sql_query("""
SELECT
    t.team_name AS team,
    p.player_name AS player,
    SUM(f.runs_off_bat) AS runs,
    SUM(f.is_boundary) AS boundaries
FROM fact_ball_delivery f
JOIN dim_team t
    ON f.batting_team_key = t.team_key
JOIN dim_player p
    ON f.striker_key = p.player_key
WHERE t.team_name = ?
GROUP BY t.team_name, p.player_name
ORDER BY runs DESC
LIMIT 10;
""", conn, params=[top_team])

print("Top team:", top_team)
print(drilldown.to_string(index=False))


# 3. SLICE: One dimension value
print("\n3. SLICE - Gujarat Titans")

slice_result = pd.read_sql_query("""
SELECT
    t.team_name AS team,
    SUM(f.total_runs) AS total_runs,
    SUM(f.is_boundary) AS boundaries,
    SUM(f.is_wicket) AS wickets_lost
FROM fact_ball_delivery f
JOIN dim_team t
    ON f.batting_team_key = t.team_key
WHERE t.team_name = 'Gujarat Titans'
GROUP BY t.team_name;
""", conn)

print(slice_result.to_string(index=False))


# 4. DICE: Multiple dimension filters
print("\n4. DICE - Selected Teams and Innings 1/2")

dice_result = pd.read_sql_query("""
SELECT
    t.team_name AS team,
    f.innings,
    SUM(f.total_runs) AS total_runs,
    SUM(f.is_boundary) AS boundaries
FROM fact_ball_delivery f
JOIN dim_team t
    ON f.batting_team_key = t.team_key
WHERE t.team_name IN (
    'Gujarat Titans',
    'Rajasthan Royals',
    'Royal Challengers Bengaluru'
)
AND f.innings IN (1, 2)
GROUP BY t.team_name, f.innings
ORDER BY t.team_name, f.innings;
""", conn)

print(dice_result.to_string(index=False))


# 5. PIVOT: Team vs Innings 1 and 2 only
print("\n5. PIVOT - Team vs Innings 1 and 2")

pivot_base = pd.read_sql_query("""
SELECT
    t.team_name AS team,
    f.innings,
    SUM(f.total_runs) AS total_runs
FROM fact_ball_delivery f
JOIN dim_team t
    ON f.batting_team_key = t.team_key
WHERE f.innings IN (1, 2)
GROUP BY t.team_name, f.innings;
""", conn)

pivot = pivot_base.pivot_table(
    index="team",
    columns="innings",
    values="total_runs",
    aggfunc="sum",
    fill_value=0
).reindex(columns=[1, 2], fill_value=0).reset_index()

pivot.columns = [
    "team",
    "innings_1_runs",
    "innings_2_runs"
]

print(pivot.to_string(index=False))


# Close database connection
conn.close()

print("\n" + "=" * 75)
print("OLAP ANALYSIS COMPLETE")
print("=" * 75)
