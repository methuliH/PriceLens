import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

_client: AsyncIOMotorClient | None = None
_DB_NAME = os.getenv("DB_NAME", "pricelens")


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        uri = os.getenv("MONGO_URI")
        if not uri:
            raise RuntimeError("MONGO_URI not set in environment")
        _client = AsyncIOMotorClient(uri)
    return _client


def get_db():
    return get_client()[_DB_NAME]


def listings():
    return get_db()["listings"]


def scrape_runs():
    return get_db()["scrape_runs"]
