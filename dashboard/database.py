"""
SQLite database module for OpenLitterPI dashboard.
Stores events (state changes, cycles, homing results) with automatic rotation.
"""

import json
import sqlite3
import time
import os

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'openlitterpi.db')

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  REAL    NOT NULL,
    event_type TEXT    NOT NULL,
    detail     TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
"""


def get_connection(db_path=None):
    """Get a SQLite connection with WAL mode for concurrent read/write."""
    path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


def insert_events(conn, events):
    """
    Insert a batch of event dicts into the database.

    Each event dict has: event_type, detail (dict or None), timestamp (optional).
    """
    rows = []
    for event in events:
        rows.append((
            event.get('timestamp', time.time()),
            event['event_type'],
            json.dumps(event.get('detail')) if event.get('detail') else None,
        ))
    conn.executemany(
        "INSERT INTO events (timestamp, event_type, detail) VALUES (?, ?, ?)",
        rows
    )
    conn.commit()


def rotate(conn, max_age_days=30):
    """Delete events older than max_age_days. Returns number of rows deleted."""
    cutoff = time.time() - (max_age_days * 24 * 3600)
    cursor = conn.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
    conn.commit()
    return cursor.rowcount


def get_recent_events(conn, limit=20):
    """Get the most recent events, newest first."""
    cursor = conn.execute(
        "SELECT timestamp, event_type, detail FROM events ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    result = []
    for ts, event_type, detail in rows:
        result.append({
            'timestamp': ts,
            'event_type': event_type,
            'detail': json.loads(detail) if detail else {},
        })
    return result


def get_cycle_stats(conn):
    """Get cycle statistics: total, last cycle time, average duration."""
    cursor = conn.execute(
        "SELECT COUNT(*), AVG(json_extract(detail, '$.duration_seconds')), "
        "MAX(timestamp) FROM events WHERE event_type = 'cycle_complete'"
    )
    row = cursor.fetchone()
    return {
        'total_cycles': row[0] or 0,
        'avg_duration': round(row[1], 1) if row[1] else 0,
        'last_cycle_time': row[2],
    }


def get_detection_stats_today(conn):
    """Get today's detection count and usage minutes."""
    today_start = time.time() - (time.time() % 86400)
    cursor = conn.execute(
        "SELECT COUNT(*) FROM events WHERE event_type = 'state_change' "
        "AND json_extract(detail, '$.new') = 'DETECTED' AND timestamp >= ?",
        (today_start,)
    )
    detections = cursor.fetchone()[0]

    cursor = conn.execute(
        "SELECT COUNT(*) FROM events WHERE event_type = 'state_change' "
        "AND json_extract(detail, '$.new') = 'USING' AND timestamp >= ?",
        (today_start,)
    )
    usages = cursor.fetchone()[0]

    return {
        'detections_today': detections,
        'usages_today': usages,
    }


def get_homing_stats(conn, limit=5):
    """Get the last N homing results."""
    cursor = conn.execute(
        "SELECT timestamp, detail FROM events WHERE event_type = 'homing_result' "
        "ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    )
    results = []
    for ts, detail in cursor.fetchall():
        d = json.loads(detail) if detail else {}
        results.append({
            'timestamp': ts,
            'aligned': d.get('aligned', False),
            'final_error_px': d.get('final_error_px', 0),
            'attempts': d.get('attempts', 0),
        })
    return results
