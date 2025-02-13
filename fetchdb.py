import os
import json
from pymongo import MongoClient
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file.
    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        print("MONGO_URI not found in environment.")
        return
    # Connect to MongoDB client
    client = MongoClient(MONGO_URI)
    all_data = []
    # List all database names
    db_names = client.list_database_names()
    # Filter out system databases (you can adjust this filter as needed)
    system_dbs = {"admin", "config", "local"}
    for db_name in db_names:
        if db_name in system_dbs:
            continue
        # Optionally, if analysis databases use a specific prefix (e.g. "db_"), uncomment:
        # if not db_name.startswith("db_"):
        #     continue
        db = client[db_name]
        # Check if the "analysis" collection exists
        if "analysis" in db.list_collection_names():
            collection = db.analysis
            docs = list(collection.find())
            # Convert ObjectId values to strings for JSON serialization and add db name for context.
            for doc in docs:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
                doc["db_name"] = db_name
            all_data.extend(docs)
    # Write the fetched data to a file
    output_file = "database_data.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        # Ensure_ascii=False to preserve non-ASCII characters (e.g., Vietnamese language)
        json.dump(all_data, f, indent=4, default=str, ensure_ascii=False)
    print(f"Fetched {len(all_data)} documents and saved to {output_file}")
if __name__ == "__main__":
    main()
