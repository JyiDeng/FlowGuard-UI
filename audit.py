# -*- coding: utf-8 -*-
"""Audit logging."""
import uuid
from flask import request

from db import get_db, utc_now_iso
from security import is_path_traversal


def new_request_id():
    return str(uuid.uuid4())


def log_audit(user, action, status, decision=None, resource_key=None, message=None):
    user_id = user["id"] if user else None
    username = user["username"] if user else None
    path = request.path
    full_path = request.full_path

    with get_db() as conn:
        conn.execute(
            "INSERT INTO audit_logs (ts, request_id, user_id, username, action, method, path, full_path, status, ip, user_agent, is_path_traversal, decision, resource_key, message) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                utc_now_iso(),
                request.headers.get("X-Request-Id") or new_request_id(),
                user_id,
                username,
                action,
                request.method,
                path,
                full_path,
                int(status),
                request.remote_addr,
                request.headers.get("User-Agent"),
                1 if is_path_traversal(path) else 0,
                decision,
                resource_key,
                message,
            ),
        )
