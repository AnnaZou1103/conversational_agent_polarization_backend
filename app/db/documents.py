import os
from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Connection
uri = os.getenv("MONGODB_URI")
client = MongoClient(uri)

# Create db and collection at first insertion
db = client["study_db"]
user_docs = db["users"]
conversation_docs = db["conversations"]
conversations_archive = db["conversations_archive"]

conversation_docs.create_index("study_id", unique=True)
conversations_archive.create_index("study_id")
