# -*- coding: utf-8 -*-
"""Simple fixed-window rate limiting backed by SQLite."""
import os
from datetime import datetime, timedelta, timezone

from db import get_db, utc_now_iso

RATE_LIMIT_PER_MINUTE = int(os.environ.get("FLOWGUARD_RATE_LIMIT", "60"))


def _window_start():
    now = datetime.now(timezone.utc)
    window = now.replace(second=0, microsecond=0)
    return window


def check_rate_limit(limit_key):
    window = _window_start()
    window_iso = window.isoformat()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, count FROM rate_limits WHERE limit_key = ? AND window_start = ?",
            (limit_key, window_iso),
        )
        row = cur.fetchone()
        if not row:
            cur.execute(
                "INSERT INTO rate_limits (limit_key, window_start, count) VALUES (?, ?, 1)",
                (limit_key, window_iso),
            )
            return True, RATE_LIMIT_PER_MINUTE - 1

        count = row["count"] + 1
        cur.execute("UPDATE rate_limits SET count = ? WHERE id = ?", (count, row["id"]))

        remaining = RATE_LIMIT_PER_MINUTE - count
        if remaining < 0:
            return False, 0
        return True, remaining
