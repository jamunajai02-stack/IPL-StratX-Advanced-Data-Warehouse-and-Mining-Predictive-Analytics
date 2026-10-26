
import os
from urllib.parse import quote_plus
from pymongo import MongoClient

# MongoDB Atlas credentials
username = "jamunajai02_db_user"
password = os.getenv("MONGODB_PASSWORD")

# Check that the password is available
if not password:
    raise ValueError("Set the MONGODB_PASSWORD environment variable first.")

# Encode credentials for the connection URL
username = quote_plus(username)
password = quote_plus(password)

# Create the Atlas connection string
uri = (
    f"mongodb+srv://{username}:{password}"
    "@ipl.1smt2su.mongodb.net/?retryWrites=true&w=majority"
)

try:
    # Connect to MongoDB Atlas
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)

    # Test the connection
    client.admin.command("ping")
    print("Successfully connected to MongoDB Atlas!")

    # Select the IPL StratX database
    db = client["IPL_StratX"]
    print("Database:", db.name)

except Exception as e:
    print("Connection error:", e)

finally:
    if "client" in locals():
        client.close()
