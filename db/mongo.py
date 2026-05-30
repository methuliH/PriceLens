import os
import datetime import datetime,timezone
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import mongo_client
from dotenv import load_dotenv

load_dotenv()

_client: AsyncIOMotorClient | None = None

def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        uri = os.getenv("MONGO_URI")
        if not uri:
            raise RuntimeError("MONGO_URI not set in environment")
        _client = AsyncIOMotorClient(uri)
    return _client
 
 
def get_db():
    return get_client()["riyaprice"]
 
 
def listings():
    return get_db()["listings"]
 
 
def scrape_runs():
    return get_db()["scrape_runs"]