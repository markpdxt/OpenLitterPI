"""
Flask web dashboard for OpenLitterPI.
"""

import time
from datetime import datetime
from flask import Flask, render_template, jsonify

from dashboard import database


def create_app(event_logger, db_path=None):
    """Create Flask app with routes bound to the given EventLogger."""
    app = Flask(__name__)

    conn = database.get_connection(db_path)

    @app.template_filter('timestamp')
    def format_timestamp(ts):
        if ts is None:
            return 'Never'
        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    @app.template_filter('duration')
    def format_duration(seconds):
        if not seconds:
            return '0s'
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f'{h}h {m}m {s}s'
        if m:
            return f'{m}m {s}s'
        return f'{s}s'

    @app.route('/')
    def index():
        live = event_logger.live_state
        uptime = time.time() - live.get('uptime_start', time.time())
        cycle_stats = database.get_cycle_stats(conn)
        detection_stats = database.get_detection_stats_today(conn)
        homing_stats = database.get_homing_stats(conn)
        recent = database.get_recent_events(conn)
        return render_template('dashboard.html',
                               live=live,
                               uptime=uptime,
                               cycle_stats=cycle_stats,
                               detection_stats=detection_stats,
                               homing_stats=homing_stats,
                               recent=recent)

    @app.route('/api/status')
    def api_status():
        return jsonify(event_logger.live_state)

    return app
