"""
Event logger for OpenLitterPI dashboard.

Provides a non-blocking interface for detect.py to log events and share
live state with the Flask dashboard. Events are queued and written to
SQLite in batches by a background thread.
"""

import queue
import threading
import time

from dashboard import database
from notifications import send_message


STALE_CYCLE_HOURS = 12
STALE_CYCLE_NOTIFY_EMAIL = "notify@pdxt.com"


class EventLogger:
    def __init__(self, db_path=None, rotation_days=30):
        self._db_path = db_path
        self._rotation_days = rotation_days
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._live_state = {
            'status': 'IDLE',
            'elapsed_time': 0,
            'fps': 0.0,
            'occupied_frames': 0,
            'uptime_start': time.time(),
        }
        self._conn = None
        self._thread = None
        self._running = False
        self._last_rotation = time.time()
        self._last_stale_check = 0
        self._stale_notified = False

    def start(self):
        """Start the background DB writer thread."""
        self._conn = database.get_connection(self._db_path)
        self._running = True
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the writer thread and flush remaining events."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self._drain()
        if self._conn:
            self._conn.close()

    def log_event(self, event_type, detail=None):
        """Non-blocking event log. Called from detect.py main loop."""
        self._queue.put({
            'timestamp': time.time(),
            'event_type': event_type,
            'detail': detail,
        })

    def update_live_state(self, status, elapsed_time, fps, occupied_frames):
        """Update shared live state dict. Called from detect.py."""
        with self._lock:
            self._live_state['status'] = status
            self._live_state['elapsed_time'] = elapsed_time
            self._live_state['fps'] = fps
            self._live_state['occupied_frames'] = occupied_frames

    @property
    def live_state(self):
        """Read live state. Called from Flask routes."""
        with self._lock:
            return dict(self._live_state)

    def _writer_loop(self):
        """Background thread: drain queue every second, rotate hourly."""
        while self._running:
            time.sleep(1.0)
            self._drain()
            now = time.time()
            # Check rotation hourly
            if now - self._last_rotation > 3600:
                database.rotate(self._conn, self._rotation_days)
                self._last_rotation = now
            # Check for stale cycle hourly
            if now - self._last_stale_check > 3600:
                self._check_stale_cycle()
                self._last_stale_check = now

    def _check_stale_cycle(self):
        """Send email alert if last cycle was more than 12 hours ago."""
        stats = database.get_cycle_stats(self._conn)
        last_cycle = stats.get('last_cycle_time')
        if last_cycle is None:
            return
        hours_since = (time.time() - last_cycle) / 3600
        if hours_since > STALE_CYCLE_HOURS and not self._stale_notified:
            try:
                send_message(
                    f"No cycle in {hours_since:.1f} hours",
                    photo=None,
                    include_image=False,
                    to=STALE_CYCLE_NOTIFY_EMAIL,
                )
                self._stale_notified = True
                print(f"Stale cycle alert sent: {hours_since:.1f}h since last cycle")
            except Exception as e:
                print(f"Failed to send stale cycle alert: {e}")
        elif hours_since <= STALE_CYCLE_HOURS:
            self._stale_notified = False

    def _drain(self):
        """Drain all queued events into the database."""
        batch = []
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if batch and self._conn:
            database.insert_events(self._conn, batch)
