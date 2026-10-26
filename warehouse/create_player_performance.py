import sqlite3
import pandas as pd

conn = sqlite3.connect("outputs/IPL_StratX_warehouse.db")
q = """
SELECT p.player_name, f.runs, f.strike_rate, f.wickets, f.economy, f.matches
FROM fact_player_performance f
JOIN dim_player p ON p.player_key = f.player_key
ORDER BY f.runs DESC
LIMIT 10
"""
print("TOP 10 PLAYERS BY RUNS")
print(pd.read_sql_query(q, conn).to_string(index=False))
conn.close()