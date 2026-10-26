
import os
import pandas as pd

# ============================================================
# IPL STRATX - COMBINED DATA PREPROCESSING (2008-2026)
# ============================================================

INPUT_FILE = "outputs/IPL_StratX_combined_2008_2026.csv"
OUTPUT_FILE = "outputs/IPL_StratX_cleaned_2008_2026.csv"

os.makedirs("outputs", exist_ok=True)

print("=" * 65)
print("IPL STRATX - COMBINED DATA PREPROCESSING")
print("=" * 65)

# 1. LOAD DATA
print("\n1. Loading combined dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")
print("Original rows:", len(df))
print("Original columns:", len(df.columns))

# 2. DISPLAY COLUMNS
print("\n2. Dataset columns:")
print(df.columns.tolist())

# 3. REMOVE EXACT DUPLICATES
print("\n3. Checking duplicate records...")

duplicates = df.duplicated().sum()
print("Duplicate records found:", duplicates)

df = df.drop_duplicates().copy()

print("Rows after duplicate removal:", len(df))

# 4. CONVERT DATE
print("\n4. Converting date column...")

if "start_date" in df.columns:
    df["start_date"] = pd.to_datetime(
        df["start_date"],
        errors="coerce"
    )

# 5. CONVERT NUMERIC COLUMNS
print("\n5. Converting numeric columns...")

numeric_columns = [
    "season",
    "match_id",
    "innings",
    "runs_off_bat",
    "extras",
    "wides",
    "noballs",
    "byes",
    "legbyes",
    "penalty",
    "total_runs",
    "is_wicket",
    "is_boundary",
    "legal_delivery",
    "is_super_over",
    "over_number",
    "ball_number"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

# Fill missing values only in the basic run components.
# Do not fill historical-only features with fabricated values.
run_columns = [
    "runs_off_bat",
    "extras",
    "wides",
    "noballs",
    "byes",
    "legbyes",
    "penalty"
]

for col in run_columns:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# 6. TOTAL RUNS
print("\n6. Checking total runs...")

calculated_runs = (
    df["runs_off_bat"] + df["extras"]
)

if "total_runs" not in df.columns:
    df["total_runs"] = calculated_runs
else:
    # Keep existing historical totals; calculate only missing values.
    df["total_runs"] = df["total_runs"].fillna(
        calculated_runs
    )

# 7. WICKET INDICATOR
print("\n7. Checking wicket indicator...")

if "wicket_type" in df.columns:
    wicket_type = (
        df["wicket_type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    wicket_from_type = (
        wicket_type.ne("") &
        wicket_type.ne("none") &
        wicket_type.ne("nan")
    ).astype(int)
else:
    wicket_from_type = pd.Series(
        0, index=df.index, dtype=int
    )

if "is_wicket" not in df.columns:
    df["is_wicket"] = wicket_from_type
else:
    df["is_wicket"] = df["is_wicket"].fillna(
        wicket_from_type
    )

# 8. BOUNDARY INDICATOR
print("\n8. Checking boundary indicator...")

boundary_from_runs = (
    df["runs_off_bat"].isin([4, 6])
).astype(int)

if "is_boundary" not in df.columns:
    df["is_boundary"] = boundary_from_runs
else:
    df["is_boundary"] = df["is_boundary"].fillna(
        boundary_from_runs
    )

# 9. LEGAL DELIVERY INDICATOR
print("\n9. Checking legal delivery indicator...")

legal_from_extras = (
    (df["wides"] == 0) &
    (df["noballs"] == 0)
).astype(int)

if "legal_delivery" not in df.columns:
    df["legal_delivery"] = legal_from_extras
else:
    df["legal_delivery"] = df["legal_delivery"].fillna(
        legal_from_extras
    )

# 10. CHECK IMPORTANT FIELDS
print("\n10. Validating important fields...")

for col in ["season", "match_id", "total_runs"]:
    if col in df.columns:
        print(f"Missing {col} values:", df[col].isna().sum())

if "total_runs" in df.columns:
    print(
        "Negative total-run records:",
        (df["total_runs"] < 0).sum()
    )

if "season" in df.columns:
    print("Seasons included:")
    print(
        sorted(df["season"].dropna().unique().tolist())
    )

if "match_id" in df.columns:
    print(
        "Unique match IDs:",
        df["match_id"].nunique()
    )

# 11. CHECK HISTORICAL-ONLY FEATURES
print("\n11. Checking additional historical features...")

historical_features = [
    "batsman_type",
    "bowler_type",
    "fielders_involved",
    "is_super_over",
    "over_number",
    "ball_number"
]

for col in historical_features:
    if col in df.columns:
        print(
            f"{col}: {df[col].notna().sum()} non-missing values"
        )

# 12. MISSING VALUE SUMMARY
print("\n12. Missing values after preprocessing:")

missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) > 0:
    print(missing.sort_values(ascending=False))
else:
    print("No missing values.")

# 13. SAVE TO A NEW FILE
print("\n13. Saving cleaned combined dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Preprocessing completed successfully.")
print("Final rows:", len(df))
print("Final columns:", len(df.columns))
print("Output file:", OUTPUT_FILE)

print("\n" + "=" * 65)
print("IPL STRATX PREPROCESSING COMPLETE")
print("=" * 65)
