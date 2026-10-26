
import os
from urllib.parse import quote_plus

from pymongo import MongoClient

username = quote_plus("jamunajai02_db_user")
password = os.getenv("MONGODB_PASSWORD")

if not password:
    raise ValueError(
        "MONGODB_PASSWORD is not set. "
        "Set it in your terminal before running this file."
    )

uri = (
    f"mongodb+srv://{username}:{quote_plus(password)}"
    "@ipl.1smt2su.mongodb.net/"
    "?retryWrites=true&w=majority"
)

client = MongoClient(uri, serverSelectionTimeoutMS=20000)
client.admin.command("ping")

db = client["IPL_StratX"]
collection = db["ball_by_ball"]

print("\n===== IPL STRATX ATLAS DATA CHECK =====")

total_records = collection.count_documents({})
print(f"\nTotal ball-by-ball records: {total_records:,}")

seasons = collection.distinct("season")
seasons = sorted(
    int(float(s)) for s in seasons if s is not None
)
print(f"Seasons available: {seasons}")
print(f"Number of seasons: {len(seasons)}")

match_count = len([
    match_id
    for match_id in collection.distinct("match_id")
    if match_id is not None
])
print(f"Unique match IDs: {match_count:,}")

print("\nRecords by season:")
pipeline = [
    {
        "$group": {
            "_id": "$season",
            "records": {"$sum": 1},
            "matches": {"$addToSet": "$match_id"}
        }
    },
    {"$sort": {"_id": 1}}
]

for item in collection.aggregate(pipeline):
    matches = len([
        m for m in item["matches"] if m is not None
    ])
    print(
        f"Season {item['_id']}: "
        f"{item['records']:,} records, "
        f"{matches} matches"
    )

print("\n===== CHECK COMPLETE =====")
client.close()
