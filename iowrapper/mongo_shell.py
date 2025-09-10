from __future__ import annotations
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import os

try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None

@dataclass
class MongoConfig:
    uri: str
    db: Optional[str] = None

class MongoShell:
    def __init__(self, uri: Optional[str] = None, db: Optional[str] = None):
        load_dotenv()
        uri_env = os.getenv("MONGODB_URI")
        self.uri = uri or uri_env or "mongodb://localhost:27017"
        if MongoClient is None:
            raise ImportError("pymongo is required. Install with `pip install pymongo`.")
        self.client = MongoClient(self.uri)
        self.db_name = db

    def get_db(self, db: Optional[str] = None):
        name = db or self.db_name
        if not name:
            raise ValueError("Database name not provided.")
        return self.client[name]

    def get_collection(self, coll: str, db: Optional[str] = None):
        return self.get_db(db)[coll]

    def ping(self) -> Dict[str, Any]:
        return self.client.admin.command("ping")

    def close(self):
        self.client.close()
