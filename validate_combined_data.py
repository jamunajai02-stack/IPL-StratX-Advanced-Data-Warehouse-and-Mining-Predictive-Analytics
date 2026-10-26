
import pandas as pd
from pathlib import Path

RAW_FILE = Path("outputs/IPL_StratX_combined_2008_2026.csv")
CLEAN_FILE = Path("outputs/IPL_StratX_cleaned_2008_2026.csv")

REQUIRED_COLUMNS = [
    "season",
    "match_id",
    "innings",
    "runs_off_bat",
    "extras",
    "total_runs",
    "is_wicket",
    "is_boundary",
    "legal_delivery",
]

EXTRA_COLUMNS = [
    "batsman_type",
    "bowler_type",
    "fielders_involved",
    "is_super_over",
    "over_number",
    "ball_number",
]


def validate_dataset(path):
    print("\n" + "=" * 65)
    print("VALIDATING:", path)
    print("=" * 65)

    if not path.exists():
        print("ERROR: File not found.")
        return

    df = pd.read_csv(path, low_memory=False)

    print("\n1. DATASET SUMMARY")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Seasons:", sorted(df["season"].dropna().unique().tolist()))
    print("Unique match IDs:", df["match_id"].nunique())
    print("Exact duplicate rows:", df.duplicated().sum())

    print("\n2. REQUIRED COLUMN CHECK")
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        print("Missing columns:", missing_columns)
        return
    else:
        print("All required columns are present.")

    # Convert temporary copies for reliable numeric comparisons.
    # This does not modify the saved dataset.
    for col in REQUIRED_COLUMNS:
        if col in ["season", "match_id", "innings"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print("\n3. BASIC DATA QUALITY")
    for col in ["season", "match_id", "total_runs"]:
        print(f"Missing {col}:", df[col].isna().sum())

    print(
        "Negative total runs:",
        (df["total_runs"] < 0).sum()
    )

    print("\n4. MATCH ID CONSISTENCY")
    match_seasons = df.groupby("match_id")["season"].nunique()
    cross_season_ids = (match_seasons > 1).sum()

    print(
        "Match IDs appearing in multiple seasons:",
        cross_season_ids
    )

    print("\n5. RUN TOTAL CONSISTENCY")

    runs = pd.to_numeric(df["runs_off_bat"], errors="coerce")
    extras = pd.to_numeric(df["extras"], errors="coerce")
    total = pd.to_numeric(df["total_runs"], errors="coerce")

    comparable = runs.notna() & extras.notna() & total.notna()
    calculated = runs + extras

    run_mismatches = (
        comparable & ((calculated - total).abs() > 0)
    )

    print("Comparable records:", comparable.sum())
    print("Total-run mismatches:", run_mismatches.sum())

    print("\n6. DERIVED FEATURE CHECKS")

    for col in ["is_wicket", "is_boundary", "legal_delivery"]:
        values = pd.to_numeric(df[col], errors="coerce")
        invalid = values.notna() & ~values.isin([0, 1])

        print(
            f"{col}: missing={values.isna().sum()}, "
            f"invalid non-binary values={invalid.sum()}"
        )

    # Compare boundaries with runs scored from the bat.
    boundary = pd.to_numeric(df["is_boundary"], errors="coerce")
    boundary_comparable = runs.notna() & boundary.notna()

    boundary_mismatches = (
        boundary_comparable &
        (boundary != runs.isin([4, 6]).astype(int))
    )

    print(
        "Boundary mismatches against 4/6 bat runs:",
        boundary_mismatches.sum()
    )

    # Compare legal-delivery values when wides and no-balls are available.
    if "wides" in df.columns and "noballs" in df.columns:
        wides = pd.to_numeric(df["wides"], errors="coerce")
        noballs = pd.to_numeric(df["noballs"], errors="coerce")
        legal = pd.to_numeric(df["legal_delivery"], errors="coerce")

        legal_comparable = (
            wides.notna() & noballs.notna() & legal.notna()
        )

        expected_legal = (
            (wides == 0) & (noballs == 0)
        ).astype(int)

        legal_mismatches = (
            legal_comparable & (legal != expected_legal)
        )

        print(
            "Legal-delivery mismatches where comparable:",
            legal_mismatches.sum()
        )

    print("\n7. FEATURE COVERAGE BY SEASON GROUP")

    for group_name, subset in [
        ("Historical 2008-2025", df[df["season"] < 2026]),
        ("2026", df[df["season"] == 2026]),
    ]:
        print(f"\n{group_name}: {len(subset)} rows")

        for col in EXTRA_COLUMNS:
            if col in subset.columns:
                print(
                    f"  {col}: {subset[col].notna().sum()} "
                    "non-missing"
                )

    print("\n8. MISSING VALUE SUMMARY")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing):
        print(missing.head(15))
    else:
        print("No missing values.")

    print("\nValidation finished for:", path)


validate_dataset(RAW_FILE)
validate_dataset(CLEAN_FILE)

print("\n" + "=" * 65)
print("ALL VALIDATION CHECKS FINISHED")
print("=" * 65)
