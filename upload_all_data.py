
# IPL StratX - Safe MongoDB Atlas Dataset Uploader
import os
from urllib.parse import quote_plus

import pandas as pd
from pymongo import MongoClient

# 1. Credentials
username = quote_plus("jamunajai02_db_user")
password = os.getenv("MONGODB_PASSWORD")

if not password:
    raise ValueError(
        "MONGODB_PASSWORD is not set in the terminal."
    )

uri = (
    f"mongodb+srv://{username}:{quote_plus(password)}"
    "@ipl.1smt2su.mongodb.net/"
    "?retryWrites=true&w=majority"
)

client = MongoClient(
    uri,
    serverSelectionTimeoutMS=20000,
    connectTimeoutMS=20000,
    socketTimeoutMS=120000,
)

folder = r"D:\IPL\outputs"
db_name = "IPL_StratX"
batch_size = 1000

# 2. Expanded dataset and current mining outputs
datasets = {
    "IPL_StratX_cleaned_2008_2026.csv": "ball_by_ball",
    "IPL_StratX_classification_data.csv": "classification_data",
    "IPL_StratX_score_predictions.csv": "score_predictions",
    "IPL_StratX_feature_importance.csv": "feature_importance",
    "IPL_StratX_frequent_itemsets.csv": "frequent_itemsets",
    "IPL_StratX_association_rules.csv": "association_rules",
}

# 3. Check every file BEFORE making any collection changes
missing_files = [
    filename
    for filename in datasets
    if not os.path.isfile(os.path.join(folder, filename))
]

if missing_files:
    raise FileNotFoundError(
        "Upload cancelled. These files are missing:\n"
        + "\n".join(missing_files)
    )

staging_names = {
    target: f"_staging_{target}"
    for target in datasets.values()
}

try:
    client.admin.command("ping")
    print("Connected to MongoDB Atlas!", flush=True)

    db = client[db_name]

    # 4. Upload every dataset to temporary collections first.
    # Existing production collections remain untouched here.
    try:
        for filename, target in datasets.items():
            path = os.path.join(folder, filename)
            staging = staging_names[target]

            print(f"\nReading {filename}...", flush=True)
            df = pd.read_csv(path, low_memory=False)
            total = len(df)

            if total == 0:
                raise ValueError(
                    f"{filename} is empty. Upload cancelled."
                )

            # Convert NaN/NaT values to JSON nulls and Python types.
            records = pd.read_json(
                df.to_json(orient="records", date_format="iso"),
                orient="records",
            ).to_dict("records")

            temp_collection = db[staging]

            # This is a temporary collection only.
            temp_collection.drop()

            print(
                f"Uploading {total:,} rows to {staging}...",
                flush=True,
            )

            for start in range(0, total, batch_size):
                batch = records[start:start + batch_size]
                if batch:
                    temp_collection.insert_many(batch)

                uploaded = min(start + batch_size, total)
                print(
                    f"\r{staging}: {uploaded:,}/{total:,}",
                    end="",
                    flush=True,
                )

            actual = temp_collection.count_documents({})

            if actual != total:
                raise RuntimeError(
                    f"Count mismatch for {target}: "
                    f"expected {total}, found {actual}"
                )

            print(
                f"\nVerified staging collection: "
                f"{staging} ({actual:,} documents)",
                flush=True,
            )

    except Exception:
        # Originals are still untouched. Remove temporary collections.
        for staging in staging_names.values():
            db[staging].drop()
        raise

    # 5. All uploads verified. Replace each target collection.
    # Only the six collections listed above are replaced.
    print("\nAll staging collections verified.", flush=True)
    print("Activating the uploaded datasets...", flush=True)

    for target, staging in staging_names.items():
        db[staging].rename(
            target,
            dropTarget=True,
        )

        actual = db[target].count_documents({})
        print(
            f"Activated {target}: {actual:,} documents",
            flush=True,
        )

    print("\nAll selected datasets uploaded successfully!", flush=True)

    # 6. Final ball-by-ball verification
    ball_collection = db["ball_by_ball"]

    print("\n===== FINAL VERIFICATION =====")
    print(
        "Ball records:",
        f"{ball_collection.count_documents({}):,}",
    )
    print(
        "Distinct match IDs:",
        len([
            value
            for value in ball_collection.distinct("match_id")
            if value is not None
        ]),
    )
    print("Season counts:")

    pipeline = [
        {
            "$group": {
                "_id": "$season",
                "records": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
    ]

    for item in ball_collection.aggregate(pipeline):
        print(f"{item['_id']}: {item['records']:,} records")

finally:
    client.close()
