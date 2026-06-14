"""SQLite persistence for recent detections."""
import sqlite3
import json
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "detections.db"


def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL,
            attack_class TEXT,
            confidence REAL,
            is_attack INTEGER,
            features TEXT
        )
    """)
    con.commit()
    con.close()


def save_detection(attack_class, confidence, is_attack, features):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO detections (ts, attack_class, confidence, is_attack, features)"
        " VALUES (?, ?, ?, ?, ?)",
        (time.time(), attack_class, confidence, int(is_attack), json.dumps(features)),
    )
    con.commit()
    con.close()


def recent(limit=100):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]