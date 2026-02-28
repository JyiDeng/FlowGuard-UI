# -*- coding: utf-8 -*-
"""
Authentication and user/session management.
"""
import os
import uuid
from datetime import datetime, timedelta, timezone
from werkzeug.security import generate_password_hash, check_password_hash

from db import get_db, utc_now_iso
from security import generate_csrf_token, generate_token, verify_token

DEFAULT_ADMIN_USERNAME = os.environ.get("FLOWGUARD_ADMIN_USER", "admin")
DEFAULT_ADMIN_PASSWORD = os.environ.get("FLOWGUARD_ADMIN_PASS", "admin123")
DEFAULT_GUEST_USERNAME = os.environ.get("FLOWGUARD_GUEST_USER", "guest")
DEFAULT_GUEST_PASSWORD = os.environ.get("FLOWGUARD_GUEST_PASS", "guest123")

SESSION_TTL_SECONDS = int(os.environ.get("FLOWGUARD_TOKEN_TTL", "28800"))


def ensure_default_accounts():
    from acl import ensure_roles_permissions, assign_role_to_user

    ensure_roles_permissions()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = ?", (DEFAULT_ADMIN_USERNAME,))
        if cur.fetchone() is None:
            cur.execute(
                "INSERT INTO users (username, password_hash, is_active, attrs_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    DEFAULT_ADMIN_USERNAME,
                    generate_password_hash(DEFAULT_ADMIN_PASSWORD),
                    1,
                    '{"department": "security", "clearance": "high"}',
                    utc_now_iso(),
                ),
            )

        cur.execute("SELECT id FROM users WHERE username = ?", (DEFAULT_GUEST_USERNAME,))
        if cur.fetchone() is None:
            cur.execute(
                "INSERT INTO users (username, password_hash, is_active, attrs_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    DEFAULT_GUEST_USERNAME,
                    generate_password_hash(DEFAULT_GUEST_PASSWORD),
                    1,
                    '{"department": "public", "clearance": "low"}',
                    utc_now_iso(),
                ),
            )

    assign_role_to_user(DEFAULT_ADMIN_USERNAME, "admin")
    assign_role_to_user(DEFAULT_GUEST_USERNAME, "guest")


def create_session(user_id, ip, user_agent):
    session_id = str(uuid.uuid4())
    csrf_token = generate_csrf_token()
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=SESSION_TTL_SECONDS)

    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions (id, user_id, csrf_token, created_at, expires_at, ip, user_agent, revoked) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            (session_id, user_id, csrf_token, now.isoformat(), expires_at.isoformat(), ip, user_agent),
        )

    token = generate_token(session_id, user_id)
    return token, csrf_token, session_id


def authenticate_user(username, password):
    with get_db() as conn:
        cur = conn.execute("SELECT * FROM users WHERE username = ? AND is_active = 1", (username,))
        row = cur.fetchone()
        if not row:
            return None
        if not check_password_hash(row["password_hash"], password):
            return None
        return row


def get_session_by_id(session_id):
    with get_db() as conn:
        cur = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        return cur.fetchone()


def revoke_session(session_id):
    with get_db() as conn:
        conn.execute("UPDATE sessions SET revoked = 1 WHERE id = ?", (session_id,))


def get_user_by_id(user_id):
    with get_db() as conn:
        cur = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        return cur.fetchone()


def resolve_token(token):
    payload = verify_token(token)
    if not payload:
        return None, None
    session_id = payload.get("sid")
    user_id = payload.get("uid")
    if not session_id or not user_id:
        return None, None

    session = get_session_by_id(session_id)
    if not session or session["revoked"]:
        return None, None

    try:
        expires_at = datetime.fromisoformat(session["expires_at"])
        if expires_at < datetime.now(timezone.utc):
            return None, None
    except Exception:
        return None, None

    user = get_user_by_id(user_id)
    if not user:
        return None, None

    return user, session
