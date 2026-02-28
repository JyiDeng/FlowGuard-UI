# -*- coding: utf-8 -*-
"""
Security helpers: tokens, CSRF, anti-replay, rate limiting.
"""
import base64
import hashlib
import hmac
import os
import time
from datetime import datetime, timedelta, timezone
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from db import get_db, utc_now_iso

TOKEN_TTL_SECONDS = int(os.environ.get("FLOWGUARD_TOKEN_TTL", "28800"))
NONCE_TTL_SECONDS = int(os.environ.get("FLOWGUARD_NONCE_TTL", "300"))


def _get_secret():
    secret = os.environ.get("FLOWGUARD_SECRET_KEY", "dev-insecure-secret")
    return secret


def get_serializer():
    return URLSafeTimedSerializer(_get_secret(), salt="flowguard-auth")


def generate_token(session_id, user_id):
    payload = {"sid": session_id, "uid": user_id}
    return get_serializer().dumps(payload)


def verify_token(token):
    try:
        payload = get_serializer().loads(token, max_age=TOKEN_TTL_SECONDS)
        return payload
    except (BadSignature, SignatureExpired):
        return None


def generate_csrf_token():
    return base64.urlsafe_b64encode(os.urandom(24)).decode("ascii")


def hash_nonce(nonce, session_id):
    data = (nonce + ":" + session_id).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def check_and_store_nonce(session_id, nonce, timestamp):
    now = int(time.time())
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        return False, "invalid timestamp"

    if abs(now - ts) > NONCE_TTL_SECONDS:
        return False, "stale request"

    nonce_hash = hash_nonce(nonce, session_id)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=NONCE_TTL_SECONDS)

    with get_db() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO request_nonces (session_id, nonce_hash, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (session_id, nonce_hash, utc_now_iso(), expires_at.isoformat()),
            )
        except Exception:
            return False, "replay detected"

    return True, None


def verify_csrf(session_row, csrf_token):
    if session_row is None:
        return False
    return hmac.compare_digest(session_row["csrf_token"], csrf_token or "")


def is_path_traversal(path):
    lowered = path.lower()
    return ".." in lowered or "%2e%2e" in lowered or "%2f" in lowered


def hash_api_key(raw_key):
    secret = _get_secret().encode("utf-8")
    return hmac.new(secret, raw_key.encode("utf-8"), hashlib.sha256).hexdigest()
