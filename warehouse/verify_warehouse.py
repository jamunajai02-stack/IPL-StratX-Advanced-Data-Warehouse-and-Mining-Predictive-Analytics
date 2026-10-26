import sqlite3
import pandas as pd

DB_FILE = "outputs/IPL_StratX_warehouse.db"
conn = sqlite3.connect(DB_FILE)

print("=" * 65)
print("IPL STRATX - WAREHOUSE VERIFICATION")
print("=" * 65)

print("\nTables:")
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn)
print(tables.to_string(index=False))

print("\nRecord counts:")
for table in tables["name"]:
    n = pd.read_sql_query(f"SELECT COUNT(*) AS count FROM {table}", conn)["count"][0]
    print(f"{table:28s}: {n}")

print("\nSample fact records:")
q = """
SELECT f.ball_key, m.match_name, t.team_name AS batting_team,
       p.player_name AS striker, f.innings, f.ball, f.total_runs
FROM fact_ball_delivery f
JOIN dim_match m ON f.match_key = m.match_key
JOIN dim_team t ON f.batting_team_key = t.team_key
JOIN dim_player p ON f.striker_key = p.player_key
LIMIT 5
"""
print(pd.read_sql_query(q, conn).to_string(index=False))

conn.close()
print("\nVERIFICATION COMPLETE")