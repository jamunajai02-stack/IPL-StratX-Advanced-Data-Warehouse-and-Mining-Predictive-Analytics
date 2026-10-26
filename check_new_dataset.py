
import pandas as pd
import os

old_file = "outputs/IPL_StratX_cleaned.csv"
new_file = "data/ball_by_ball_data.csv"

for file in [old_file, new_file]:
    print("\n" + "=" * 60)
    print("FILE:", file)

    if not os.path.exists(file):
        print("File not found!")
        continue

    # Read only the first 5 rows initially
    df = pd.read_csv(file, nrows=5, low_memory=False)

    # Count all rows without loading the whole dataset into memory
    with open(file, "rb") as f:
        row_count = sum(1 for _ in f) - 1

    print("Total rows:", row_count)
    print("Total columns:", len(df.columns))
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.to_string(index=False))
