# -*- coding: utf-8 -*-
"""
Database utilities and initialization.
"""
import os
import sqlite3
import hashlib
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.environ.get("FLOWGUARD_DB_PATH", os.path.join(os.path.dirname(__file__), "data", "app.db"))

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        is_active INTEGER NOT NULL DEFAULT 1,
        attrs_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS roles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS permissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS role_permissions (
        role_id INTEGER NOT NULL,
        permission_id INTEGER NOT NULL,
        UNIQUE(role_id, permission_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_roles (
        user_id INTEGER NOT NULL,
        role_id INTEGER NOT NULL,
        UNIQUE(user_id, role_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS resources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        resource_key TEXT UNIQUE NOT NULL,
        resource_type TEXT NOT NULL,
        name TEXT NOT NULL,
        policy_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS resource_acl (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        resource_id INTEGER NOT NULL,
        user_id INTEGER,
        role_id INTEGER,
        allow INTEGER NOT NULL DEFAULT 1
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        example_key TEXT NOT NULL,
        current_state TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        message TEXT NOT NULL,
        timestamp REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS flowchart (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alias TEXT UNIQUE NOT NULL,
        content TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS five_tuples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        flowchart TEXT NOT NULL,
        current_state TEXT NOT NULL,
        next_state TEXT NOT NULL,
        user_input TEXT NOT NULL,
        robot_output TEXT NOT NULL,
        lang TEXT NOT NULL DEFAULT 'zh'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS transition_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        example_key TEXT NOT NULL,
        action_type TEXT NOT NULL,
        from_state TEXT,
        to_state TEXT,
        trigger TEXT,
        message_id INTEGER,
        timestamp REAL NOT NULL,
        payload TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_transition_events_conversation ON transition_events (conversation_id, id)",
    "CREATE INDEX IF NOT EXISTS idx_transition_events_user ON transition_events (user_id, id)",
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        csrf_token TEXT NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        ip TEXT,
        user_agent TEXT,
        revoked INTEGER NOT NULL DEFAULT 0,
        current_conversation_id INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS request_nonces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        nonce_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        UNIQUE(session_id, nonce_hash)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS api_keys (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        key_hash TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        revoked INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS rate_limits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        limit_key TEXT NOT NULL,
        window_start TEXT NOT NULL,
        count INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        request_id TEXT NOT NULL,
        user_id INTEGER,
        username TEXT,
        action TEXT NOT NULL,
        method TEXT NOT NULL,
        path TEXT NOT NULL,
        full_path TEXT NOT NULL,
        status INTEGER NOT NULL,
        ip TEXT,
        user_agent TEXT,
        is_path_traversal INTEGER NOT NULL,
        decision TEXT,
        resource_key TEXT,
        message TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS "session" (
        user_id INTEGER NOT NULL,
        session_id TEXT PRIMARY KEY,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        client_tz TEXT,
        client_meta TEXT,
        experiment_ver TEXT,
        status TEXT NOT NULL DEFAULT 'running',
        is_success INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dialogue (
        user_id INTEGER NOT NULL,
        dialogue_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES "session"(session_id),
        conversation_id INTEGER,
        flowchart_alias TEXT,
        task_key TEXT,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        full_path TEXT,
        goal_state_id TEXT,
        path_node_count INTEGER NOT NULL DEFAULT 0,
        is_success INTEGER NOT NULL DEFAULT 0,
        meta TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dialogue_session ON dialogue (session_id, started_at)",
    "CREATE INDEX IF NOT EXISTS idx_dialogue_conversation ON dialogue (conversation_id, started_at)",
    """
    CREATE TABLE IF NOT EXISTS "turn" (
        user_id INTEGER NOT NULL,
        turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
        dialogue_id TEXT NOT NULL REFERENCES dialogue(dialogue_id),
        turn_index INTEGER NOT NULL,
        ts_start TEXT NOT NULL,
        ts_end TEXT,
        cur_state_id TEXT,
        next_state_id TEXT,
        user_utterance TEXT,
        system_reply TEXT,
        is_clarification INTEGER NOT NULL DEFAULT 0,
        is_backtrack INTEGER NOT NULL DEFAULT 0,
        is_violation INTEGER NOT NULL DEFAULT 0,
        meta TEXT,
        UNIQUE (dialogue_id, turn_index)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_turn_dialogue ON \"turn\" (dialogue_id, turn_index)",
]


@contextmanager
def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Improve concurrency and reduce lock errors
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
    except sqlite3.Error:
        # If PRAGMA fails, continue with defaults
        pass
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        cur = conn.cursor()
        for stmt in SCHEMA:
            cur.execute(stmt)
        _ensure_five_tuples_lang_column(conn)
        _ensure_study_analytics_schema(conn)
        _migrate_flowchart_storage(conn)


def _migrate_flowchart_storage(conn):
    """
    Normalize legacy five_tuples.flowchart values:
    old rows may store raw PlantUML text, new rows store flowchart alias.
    """
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT flowchart
            FROM five_tuples
            WHERE flowchart LIKE '@startuml%'
            """
        ).fetchall()
    except sqlite3.Error:
        return

    for row in rows:
        raw_flowchart = row["flowchart"]
        if not raw_flowchart:
            continue
        alias = f"legacy_{hashlib.sha1(raw_flowchart.encode('utf-8')).hexdigest()[:12]}"
        conn.execute(
            """
            INSERT INTO flowchart (alias, content)
            VALUES (?, ?)
            ON CONFLICT(alias) DO UPDATE SET content = excluded.content
            """,
            (alias, raw_flowchart),
        )
        conn.execute(
            "UPDATE five_tuples SET flowchart = ? WHERE flowchart = ?",
            (alias, raw_flowchart),
        )


def _ensure_five_tuples_lang_column(conn):
    """
    Backward-compatible migration for existing databases.
    """
    try:
        cols = conn.execute("PRAGMA table_info(five_tuples)").fetchall()
    except sqlite3.Error:
        return
    col_names = {row["name"] for row in cols}
    if "lang" not in col_names:
        conn.execute("ALTER TABLE five_tuples ADD COLUMN lang TEXT NOT NULL DEFAULT 'zh'")


def _ensure_column(conn, table_name, col_name, ddl):
    try:
        cols = conn.execute(f"PRAGMA table_info(\"{table_name}\")").fetchall()
    except sqlite3.Error:
        return
    col_names = {row["name"] for row in cols}
    if col_name not in col_names:
        conn.execute(f"ALTER TABLE \"{table_name}\" ADD COLUMN {ddl}")


def _ensure_study_analytics_schema(conn):
    """
    Forward-compatible migrations for experiment/session analytics tables.
    """
    for table_name, col_name, ddl in [
        ("session", "client_tz", "client_tz TEXT"),
        ("session", "client_meta", "client_meta TEXT"),
        ("session", "experiment_ver", "experiment_ver TEXT"),
        ("session", "status", "status TEXT NOT NULL DEFAULT 'running'"),
        ("session", "is_success", "is_success INTEGER NOT NULL DEFAULT 0"),
        ("dialogue", "conversation_id", "conversation_id INTEGER"),
        ("dialogue", "flowchart_alias", "flowchart_alias TEXT"),
        ("dialogue", "task_key", "task_key TEXT"),
        ("dialogue", "full_path", "full_path TEXT"),
        ("dialogue", "goal_state_id", "goal_state_id TEXT"),
        ("dialogue", "path_node_count", "path_node_count INTEGER NOT NULL DEFAULT 0"),
        ("dialogue", "is_success", "is_success INTEGER NOT NULL DEFAULT 0"),
        ("dialogue", "meta", "meta TEXT"),
        ("turn", "is_clarification", "is_clarification INTEGER NOT NULL DEFAULT 0"),
        ("turn", "is_backtrack", "is_backtrack INTEGER NOT NULL DEFAULT 0"),
        ("turn", "is_violation", "is_violation INTEGER NOT NULL DEFAULT 0"),
        ("turn", "meta", "meta TEXT"),
    ]:
        _ensure_column(conn, table_name, col_name, ddl)


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()
