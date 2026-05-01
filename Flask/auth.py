import os
from datetime import datetime
from typing import Optional

import bcrypt
from pymongo import MongoClient, ASCENDING


MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.environ.get("MONGODB_DB", "plant_disease")
MONGODB_COLLECTION = os.environ.get("MONGODB_COLLECTION", "users")


_client: Optional[MongoClient] = None


def _get_collection():
    global _client
    if _client is None:
        _client = MongoClient(MONGODB_URI)
    db = _client[MONGODB_DB]
    collection = db[MONGODB_COLLECTION]
    # Ensure unique email.
    collection.create_index([("email", ASCENDING)], unique=True)
    return collection


def hash_password(password: str) -> str:
    # bcrypt uses random salt each time.
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def get_user_by_email(email: str) -> Optional[dict]:
    email = email.strip().lower()
    return _get_collection().find_one({"email": email})


def create_user(username: str, email: str, password: str) -> dict:
    username = username.strip()
    email = email.strip().lower()
    if not username:
        raise ValueError("Username is required.")
    if not email:
        raise ValueError("Email is required.")
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    users = _get_collection()
    if users.find_one({"email": email}) is not None:
        raise ValueError("Email already exists.")

    user_doc = {
        "username": username,
        "email": email,
        "password_hash": hash_password(password),
        "created_at": datetime.utcnow(),
    }
    result = users.insert_one(user_doc)
    user_doc["_id"] = result.inserted_id
    return user_doc

