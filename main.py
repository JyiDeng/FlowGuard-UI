# -*- coding: utf-8 -*-
"""
FlowGuard 主程序
基于Flask的对话式流程图交互系统（含权限与审计）
"""
from flask import Flask, render_template, request, jsonify, g
import time
import json
import os
import re
import uuid
import urllib.request
import urllib.error
from datetime import datetime, timezone
from difflib import get_close_matches

# 导入自定义模块
from parser import run
import config
from examples import (
    get_example,
    get_all_examples,
    DEFAULT_EXAMPLE_KEY,
    VISIBLE_UML_EXAMPLE_KEYS,
    get_user_test_tasks,
    normalize_lang,
)
from ui_texts import get_ui_texts_bundle
from models import get_model, switch_model, get_current_model_info, ensure_model_loaded
from db import init_db, get_db, utc_now_iso
from auth import (
    ensure_default_accounts,
    authenticate_user,
    create_session,
    resolve_token,
    revoke_session,
)
from werkzeug.security import generate_password_hash
from acl import (
    ensure_roles_permissions,
    user_has_permission,
    get_allowed_example_keys,
    upsert_resource,
    ensure_resource_acls_for_defaults,
    get_resource,
)
from security import verify_csrf, check_and_store_nonce, hash_api_key
from rate_limit import check_rate_limit
from audit import log_audit

# ==================== Flask应用初始化 ====================
app = Flask(__name__)
PREFERRED_EXAMPLE_KEY = "taobao_return"


def pick_default_example(allowed_examples):
    if PREFERRED_EXAMPLE_KEY in allowed_examples:
        return PREFERRED_EXAMPLE_KEY
    return allowed_examples[0] if allowed_examples else DEFAULT_EXAMPLE_KEY


def ensure_start_state(conversation):
    if not conversation:
        return conversation
    if conversation["current_state"] in ("<end>", "stop"):
        with get_db() as conn:
            conn.execute(
                "UPDATE conversations SET current_state = ?, updated_at = ? WHERE id = ?",
                ("<start>", utc_now_iso(), conversation["id"]),
            )
        return {**conversation, "current_state": "<start>"}
    return conversation


def _is_truthy_param(value):
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def reset_conversation_state(conversation, *, trigger="system", reason="auto_reset", clear_messages=False, action_type="reset"):
    if not conversation:
        return conversation
    prev_state = conversation["current_state"]
    payload = {"reason": reason}
    with get_db() as conn:
        conn.execute(
            "UPDATE conversations SET current_state = ?, updated_at = ? WHERE id = ?",
            ("<start>", utc_now_iso(), conversation["id"]),
        )
        if clear_messages:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation["id"],))
        log_transition_event(
            conversation_id=conversation["id"],
            user_id=conversation["user_id"],
            example_key=conversation["example_key"],
            action_type=action_type,
            from_state=prev_state,
            to_state="<start>",
            trigger=trigger,
            payload=payload,
            conn=conn,
        )
    return {**conversation, "current_state": "<start>"}
app.config["SECRET_KEY"] = config.SECRET_KEY


def get_ui_lang(data=None):
    if isinstance(data, dict):
        lang = data.get("lang")
        if lang:
            return normalize_lang(lang)
    header_lang = request.headers.get("X-UI-Lang")
    if header_lang:
        return normalize_lang(header_lang)
    query_lang = request.args.get("lang")
    if query_lang:
        return normalize_lang(query_lang)
    return normalize_lang(None)


def is_study_mode_request():
    return request.headers.get("X-Study-Mode", "").strip() == "1"


def is_study_user():
    if not g.user:
        return False
    try:
        attrs = json.loads(g.user["attrs_json"] or "{}")
    except Exception:
        attrs = {}
    return attrs.get("study") == "user_test"


def can_access_study_examples_without_acl():
    return is_study_mode_request() and bool(g.user)

# ==================== 初始化与数据同步 ====================

def init_system():
    config.validate_config()
    init_db()
    ensure_roles_permissions()
    ensure_default_accounts()
    sync_resources()
    ensure_resource_acls_for_defaults()
    ensure_default_api_keys()


def sync_resources():
    examples = get_all_examples()
    for key, info in examples.items():
        upsert_resource(
            resource_key=key,
            resource_type="example",
            name=info.get("name", key),
            policy=info.get("policy", {}),
        )


def ensure_default_api_keys():
    admin_key = os.environ.get("FLOWGUARD_ADMIN_API_KEY", "admin-api-key")
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = ?", (os.environ.get("FLOWGUARD_ADMIN_USER", "admin"),))
        admin = cur.fetchone()
        if not admin:
            return
        key_hash = hash_api_key(admin_key)
        cur.execute("SELECT id FROM api_keys WHERE key_hash = ?", (key_hash,))
        if cur.fetchone() is None:
            cur.execute(
                "INSERT INTO api_keys (user_id, key_hash, name, created_at) VALUES (?, ?, ?, ?)",
                (admin["id"], key_hash, "default-admin", utc_now_iso()),
            )


init_system()

# ==================== 工具函数 ====================

def parse_plantuml_to_nodes(plantuml_code):
    """从PlantUML代码中解析节点"""
    nodes = []
    lines = plantuml_code.split('\n')
    node_id = 1

    for line in lines:
        line = line.strip()
        if line.startswith(':') and line.endswith(';'):
            node_name = line[1:-1]
            nodes.append({
                "id": node_id,
                "name": node_name,
                "type": "activity"
            })
            node_id += 1
        elif line.startswith('if (') and ') then (' in line:
            condition = line[4:line.find(') then (')]
            nodes.append({
                "id": node_id,
                "name": condition,
                "type": "decision"
            })
            node_id += 1
        elif line.startswith('repeat while'):
            start = line.find('(')
            end = line.find(')', start + 1)
            if start != -1 and end != -1 and end > start:
                condition = line[start + 1:end]
                nodes.append({
                    "id": node_id,
                    "name": condition,
                    "type": "decision"
                })
                node_id += 1

    return nodes


def build_state_mapping(plantuml_code):
    """构建状态映射"""
    plantuml_nodes = []
    lines = plantuml_code.split('\n')

    for line in lines:
        line = line.strip()
        if line.startswith(':') and line.endswith(';'):
            node_name = line[1:-1]
            plantuml_nodes.append(node_name)
        elif line.startswith('if (') and ') then (' in line:
            condition = line[4:line.find(') then (')]
            plantuml_nodes.append(condition)
        elif line.startswith('repeat while'):
            start = line.find('(')
            end = line.find(')', start + 1)
            if start != -1 and end != -1 and end > start:
                plantuml_nodes.append(line[start + 1:end])

    state_to_node_map = {
        "<start>": 0,
        "<end>": len(plantuml_nodes) + 1
    }

    for i, node_name in enumerate(plantuml_nodes):
        state_to_node_map[node_name] = i + 1

    return state_to_node_map, plantuml_nodes


def normalize_state_name(state, plantuml_code):
    """将模型返回的状态名归一到 PlantUML 中的实际节点标签。"""
    raw = str(state or "").strip()
    if not raw:
        return raw

    normalized = raw.strip("`").strip("\"'").strip()
    normalized = normalized.splitlines()[0].strip() if normalized else normalized
    normalized = re.sub(r"\s*;\s*$", "", normalized).strip()
    normalized_lower = normalized.lower()

    if normalized_lower in {"stop", "end", "<end>"}:
        return "<end>"
    if normalized_lower in {"start", "<start>"}:
        return "<start>"

    _, plantuml_nodes = build_state_mapping(plantuml_code)
    candidates = plantuml_nodes[:]

    if normalized in candidates:
        return normalized

    def _key(value):
        compact = re.sub(r"\s*;\s*$", "", str(value or "").strip().strip("`").strip("\"'")).strip()
        return re.sub(r"\s+", " ", compact).lower()

    normalized_key = _key(normalized)
    for candidate in candidates:
        if _key(candidate) == normalized_key:
            return candidate

    return normalized


def _normalize_label_key(value):
    compact = re.sub(r"\s*;\s*$", "", str(value or "").strip().strip("`").strip("\"'")).strip()
    return re.sub(r"\s+", " ", compact).lower()


def get_next_state_candidates(current_state, plantuml_code):
    """
    基于解析后的流程图，返回当前状态可到达的“下一跳”候选节点。
    会跳过 repeat 的内部 noop 节点，直接返回用户可见节点。
    """
    graph = run(plantuml_code)
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])

    outgoing = {}
    for edge in edges:
        outgoing.setdefault(edge["from"], []).append(edge["to"])

    current_key = _normalize_label_key(current_state)
    matched_ids = []
    for node_id, info in nodes.items():
        label = info.get("label") or ""
        node_type = info.get("type")
        if current_key in {"<start>", "start"} and node_type == "start":
            matched_ids.append(node_id)
        elif current_key in {"<end>", "end", "stop"} and node_type == "stop":
            matched_ids.append(node_id)
        elif _normalize_label_key(label) == current_key:
            matched_ids.append(node_id)

    def _collect_visible_successors(start_id, seen=None):
        seen = seen or set()
        if start_id in seen:
            return []
        seen.add(start_id)
        results = []
        for target_id in outgoing.get(start_id, []):
            info = nodes.get(target_id, {})
            node_type = info.get("type")
            label = info.get("label") or ""
            if node_type == "noop" or _normalize_label_key(label) == "repeat":
                results.extend(_collect_visible_successors(target_id, seen))
            else:
                results.append(label)
        return results

    candidates = []
    for node_id in matched_ids:
        candidates.extend(_collect_visible_successors(node_id))

    deduped = []
    seen_keys = set()
    for candidate in candidates:
        key = _normalize_label_key(candidate)
        if key and key not in seen_keys:
            seen_keys.add(key)
            deduped.append(candidate)
    return deduped


def resolve_next_state_from_candidates(next_state, candidates, current_state):
    """
    将模型输出约束到合法下一跳。单出口节点强制前进，多出口节点做保守模糊匹配。
    """
    if not candidates:
        return next_state

    normalized = _normalize_label_key(next_state)
    current_key = _normalize_label_key(current_state)
    candidate_map = {_normalize_label_key(candidate): candidate for candidate in candidates}

    if normalized in candidate_map:
        return candidate_map[normalized]

    if len(candidates) == 1:
        return candidates[0]

    close = get_close_matches(normalized, list(candidate_map.keys()), n=1, cutoff=0.6)
    if close:
        return candidate_map[close[0]]

    # 不再回退到 start；多出口无法判断时保持当前状态，避免误跳。
    if current_key in {"start", "<start>"}:
        return next_state
    return current_state


def to_chinese_state_for_storage(example_key, state, src_lang):
    """
    Persist states in Chinese for easier downstream analysis while keeping UI language in `lang`.
    """
    if not state:
        return state
    if normalize_lang(src_lang) == "zh":
        return state

    if state in ("<start>", "<end>", "stop"):
        return "<end>" if state == "stop" else state

    try:
        src_nodes = ["<start>"] + [n["name"] for n in parse_plantuml_to_nodes(get_example(example_key, lang=src_lang))] + ["<end>"]
        zh_nodes = ["<start>"] + [n["name"] for n in parse_plantuml_to_nodes(get_example(example_key, lang="zh"))] + ["<end>"]
    except Exception:
        return state

    if len(src_nodes) != len(zh_nodes):
        return state

    try:
        idx = src_nodes.index(state)
    except ValueError:
        return state
    return zh_nodes[idx]


def localize_state_for_ui(example_key, state, target_lang):
    """
    将会话中保存的状态名转换为当前 UI 语言对应的节点名。
    会话内优先统一存中文；这里负责在中英文视图之间映射。
    """
    if not state:
        return state

    resolved_lang = normalize_lang(target_lang)
    if state in ("<start>", "<end>", "stop"):
        return "<end>" if state == "stop" else state

    try:
        zh_nodes = ["<start>"] + [n["name"] for n in parse_plantuml_to_nodes(get_example(example_key, lang="zh"))] + ["<end>"]
        en_nodes = ["<start>"] + [n["name"] for n in parse_plantuml_to_nodes(get_example(example_key, lang="en"))] + ["<end>"]
    except Exception:
        return state

    if len(zh_nodes) != len(en_nodes):
        return state

    target_nodes = zh_nodes if resolved_lang == "zh" else en_nodes
    source_nodes = en_nodes if resolved_lang == "zh" else zh_nodes

    if state in target_nodes:
        return state

    try:
        idx = source_nodes.index(state)
        return target_nodes[idx]
    except ValueError:
        return state


def get_auth_context():
    token = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    if not token:
        token = request.headers.get("X-Auth-Token")

    user = None
    session = None
    if token:
        user, session = resolve_token(token)

    if not user:
        api_key = request.headers.get("X-API-Key")
        if api_key:
            key_hash = hash_api_key(api_key)
            with get_db() as conn:
                cur = conn.execute(
                    "SELECT u.* FROM api_keys ak JOIN users u ON ak.user_id = u.id WHERE ak.key_hash = ? AND ak.revoked = 0",
                    (key_hash,),
                )
                user = cur.fetchone()
                session = None

    return user, session


def require_permission(permission, action, resource_key=None):
    if not g.user:
        log_audit(None, action, 401, decision="unauthorized", resource_key=resource_key)
        return jsonify({"success": False, "error": "unauthorized"}), 401

    if not user_has_permission(g.user["id"], permission):
        log_audit(g.user, action, 403, decision="forbidden", resource_key=resource_key)
        return jsonify({"success": False, "error": "forbidden"}), 403

    return None


def rate_limit_guard():
    limit_key = str(g.user["id"]) if g.user else request.remote_addr
    ok, remaining = check_rate_limit(limit_key)
    if not ok:
        log_audit(g.user, "rate_limit", 429, decision="blocked")
        return jsonify({"success": False, "error": "rate limit exceeded"}), 429
    return None


def security_guards():
    if config.REQUIRE_HTTPS and not request.is_secure:
        return jsonify({"success": False, "error": "https required"}), 400

    if request.method in ["POST", "PUT", "DELETE"]:
        nonce = request.headers.get("X-Request-Nonce")
        ts = request.headers.get("X-Request-Timestamp")
        if not nonce or not ts:
            return jsonify({"success": False, "error": "missing replay protection headers"}), 400

        if g.session:
            ok, reason = check_and_store_nonce(g.session["id"], nonce, ts)
            if not ok:
                return jsonify({"success": False, "error": reason}), 400

            csrf = request.headers.get("X-CSRF-Token")
            if not verify_csrf(g.session, csrf):
                return jsonify({"success": False, "error": "invalid csrf"}), 403
        else:
            # API key calls still need anti-replay
            ok, reason = check_and_store_nonce(f"public:{request.remote_addr}", nonce, ts)
            if not ok:
                return jsonify({"success": False, "error": reason}), 400

    return None


def get_or_create_conversation(user_id, example_key):
    with get_db() as conn:
        cur = conn.execute(
            "SELECT * FROM conversations WHERE user_id = ? AND example_key = ? ORDER BY id DESC LIMIT 1",
            (user_id, example_key),
        )
        row = cur.fetchone()
        if row:
            return row

        now = utc_now_iso()
        conn.execute(
            "INSERT INTO conversations (user_id, example_key, current_state, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, example_key, "<start>", now, now),
        )
        cur = conn.execute(
            "SELECT * FROM conversations WHERE user_id = ? AND example_key = ? ORDER BY id DESC LIMIT 1",
            (user_id, example_key),
        )
        return cur.fetchone()


def set_session_conversation(session_id, conversation_id):
    with get_db() as conn:
        conn.execute(
            "UPDATE sessions SET current_conversation_id = ? WHERE id = ?",
            (conversation_id, session_id),
        )


def get_current_conversation(session_id):
    with get_db() as conn:
        cur = conn.execute("SELECT current_conversation_id FROM sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        if not row or row["current_conversation_id"] is None:
            return None
        cur = conn.execute("SELECT * FROM conversations WHERE id = ?", (row["current_conversation_id"],))
        return cur.fetchone()


def can_access_conversation(user_id, conversation_row):
    if not conversation_row:
        return False
    if conversation_row["user_id"] == user_id:
        return True
    return user_has_permission(user_id, "view_audit")


def log_transition_event(
    *,
    conversation_id,
    user_id,
    example_key,
    action_type,
    from_state=None,
    to_state=None,
    trigger=None,
    message_id=None,
    payload=None,
    conn=None,
):
    if payload is not None and not isinstance(payload, str):
        payload = json.dumps(payload, ensure_ascii=False)
    ts = time.time()
    row = (
        conversation_id,
        user_id,
        example_key,
        action_type,
        from_state,
        to_state,
        trigger,
        message_id,
        ts,
        payload,
    )
    if conn is None:
        with get_db() as local_conn:
            local_conn.execute(
                """
                INSERT INTO transition_events (
                    conversation_id, user_id, example_key, action_type,
                    from_state, to_state, trigger, message_id, timestamp, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
    else:
        conn.execute(
            """
            INSERT INTO transition_events (
                conversation_id, user_id, example_key, action_type,
                from_state, to_state, trigger, message_id, timestamp, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )


def is_stop_state_value(value):
    s = str(value or "").strip().lower()
    return s in {"stop", "<end>", "end"}


def _normalize_path_state(value):
    if not value:
        return value
    return "<end>" if str(value).strip().lower() == "stop" else value


def _json_loads_safe(value, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _json_dumps_safe(value):
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return "{}" if isinstance(value, dict) else "[]"


def _merge_json_text(old_text, patch_dict):
    base = _json_loads_safe(old_text, {})
    if not isinstance(base, dict):
        base = {"_raw": old_text}
    for k, v in (patch_dict or {}).items():
        if v is not None:
            base[k] = v
    return _json_dumps_safe(base)


def ensure_experiment_session_row(conn, data=None):
    if not (g.user and g.session):
        return None

    payload = data if isinstance(data, dict) else {}
    client_tz = str(payload.get("client_tz") or "").strip() or None
    client_meta_obj = payload.get("client_meta")
    client_meta = None
    if client_meta_obj is not None:
        client_meta = client_meta_obj if isinstance(client_meta_obj, str) else _json_dumps_safe(client_meta_obj)
    try:
        model_info = get_current_model_info()
        experiment_ver = _json_dumps_safe(model_info) if isinstance(model_info, dict) else str(model_info)
    except Exception:
        experiment_ver = str(config.MODEL_TYPE)

    conn.execute(
        """
        INSERT OR IGNORE INTO "session" (
            user_id, session_id, started_at, client_tz, client_meta, experiment_ver, status, is_success
        ) VALUES (?, ?, ?, ?, ?, ?, 'running', 0)
        """,
        (g.user["id"], g.session["id"], utc_now_iso(), client_tz, client_meta, experiment_ver),
    )
    if client_tz or client_meta or experiment_ver:
        conn.execute(
            """
            UPDATE "session"
            SET client_tz = COALESCE(?, client_tz),
                client_meta = COALESCE(?, client_meta),
                experiment_ver = COALESCE(?, experiment_ver)
            WHERE session_id = ?
            """,
            (client_tz, client_meta, experiment_ver, g.session["id"]),
        )
    cur = conn.execute('SELECT * FROM "session" WHERE session_id = ?', (g.session["id"],))
    return cur.fetchone()


def _get_active_dialogue(conn, conversation_id=None, task_key=None):
    if not (g.session and g.user):
        return None
    if conversation_id is not None:
        cur = conn.execute(
            """
            SELECT * FROM dialogue
            WHERE session_id = ? AND conversation_id = ? AND ended_at IS NULL
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (g.session["id"], conversation_id),
        )
        row = cur.fetchone()
        if row:
            return row
    if task_key:
        cur = conn.execute(
            """
            SELECT * FROM dialogue
            WHERE session_id = ? AND task_key = ? AND ended_at IS NULL
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (g.session["id"], task_key),
        )
        return cur.fetchone()
    return None


def _create_study_dialogue(conn, conversation, task_key, payload=None):
    if not (g.user and g.session):
        return None
    payload = payload or {}
    session_row = ensure_experiment_session_row(conn, payload)
    if not session_row:
        return None

    conn.execute(
        """
        UPDATE dialogue
        SET ended_at = COALESCE(ended_at, ?),
            meta = CASE
                WHEN meta IS NULL OR meta = '' THEN ?
                ELSE ?
            END
        WHERE session_id = ? AND conversation_id = ? AND ended_at IS NULL
        """,
        (
            utc_now_iso(),
            _json_dumps_safe({"closed_reason": "new_task_started"}),
            _merge_json_text("{}", {"closed_reason": "new_task_started"}),
            g.session["id"],
            conversation["id"],
        ),
    )

    dialogue_id = str(uuid.uuid4())
    start_state = _normalize_path_state(conversation["current_state"] or "<start>") or "<start>"
    meta = {
        "task_index": payload.get("task_index"),
        "total_tasks": payload.get("total_tasks"),
        "task_title": payload.get("task_title"),
        "created_from": "start_user_test_task",
    }
    try:
        conn.execute(
            """
            INSERT INTO dialogue (
                user_id, dialogue_id, session_id, conversation_id, flowchart_alias, task_key,
                started_at, full_path, goal_state_id, path_node_count, is_success, meta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, 0, ?)
            """,
            (
                g.user["id"],
                dialogue_id,
                g.session["id"],
                conversation["id"],
                task_key,
                task_key,
                utc_now_iso(),
                _json_dumps_safe([start_state]),
                1,
                _json_dumps_safe(meta),
            ),
        )
    except Exception as e:
        # Analytics logging should never block the study flow.
        print(f"Study analytics: create dialogue skipped: {e}", flush=True)
        return None
    cur = conn.execute("SELECT * FROM dialogue WHERE dialogue_id = ?", (dialogue_id,))
    return cur.fetchone()


def append_dialogue_path_and_turn(
    conn,
    *,
    conversation,
    current_state,
    next_state,
    user_message,
    system_reply,
    ts_start,
    ts_end,
):
    if not (g.user and g.session):
        return
    dialogue = _get_active_dialogue(conn, conversation_id=conversation["id"], task_key=conversation["example_key"])
    if not dialogue:
        dialogue = _create_study_dialogue(conn, conversation, conversation["example_key"], {})
    if not dialogue:
        return

    path = _json_loads_safe(dialogue["full_path"], [])
    if not isinstance(path, list):
        path = []
    cur_norm = _normalize_path_state(current_state)
    next_norm = _normalize_path_state(next_state)
    for state in [cur_norm, next_norm]:
        if not state:
            continue
        if not path or path[-1] != state:
            path.append(state)

    next_turn_index = conn.execute(
        'SELECT COALESCE(MAX(turn_index), -1) + 1 AS n FROM "turn" WHERE dialogue_id = ?',
        (dialogue["dialogue_id"],),
    ).fetchone()["n"]
    is_backtrack = int(bool(next_norm and next_norm in path[:-1]))
    conn.execute(
        """
        INSERT INTO "turn" (
            user_id, dialogue_id, turn_index, ts_start, ts_end,
            cur_state_id, next_state_id, user_utterance, system_reply,
            is_clarification, is_backtrack, is_violation, meta
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 0, ?)
        """,
        (
            g.user["id"],
            dialogue["dialogue_id"],
            next_turn_index,
            ts_start,
            ts_end,
            cur_norm,
            next_norm,
            user_message,
            system_reply,
            is_backtrack,
            _json_dumps_safe({"source": "api_chat"}),
        ),
    )
    conn.execute(
        """
        UPDATE dialogue
        SET full_path = ?, path_node_count = ?, flowchart_alias = COALESCE(flowchart_alias, ?)
        WHERE dialogue_id = ?
        """,
        (_json_dumps_safe(path), len(path), conversation["example_key"], dialogue["dialogue_id"]),
    )


def update_dialogue_goal_selection(conn, *, conversation, task_key, details=None, task_index=None):
    if not (g.user and g.session):
        return
    dialogue = _get_active_dialogue(conn, conversation_id=conversation["id"], task_key=task_key or conversation["example_key"])
    if not dialogue:
        dialogue = _create_study_dialogue(conn, conversation, task_key or conversation["example_key"], {"task_index": task_index})
    if not dialogue:
        return
    details = details if isinstance(details, dict) else {"value": details}
    goal_state_id = str(details.get("goal_state_id") or "").strip() or None
    if goal_state_id:
        try:
            goal_state_id = to_chinese_state_for_storage(task_key or conversation["example_key"], goal_state_id, get_ui_lang())
        except Exception:
            pass
    meta = _merge_json_text(
        dialogue["meta"],
        {
            "goal_selected_at": utc_now_iso(),
            "goal_node_id": details.get("node_id"),
            "goal_node_type": details.get("node_type"),
            "task_index": task_index,
        },
    )
    conn.execute(
        "UPDATE dialogue SET goal_state_id = COALESCE(?, goal_state_id), meta = ? WHERE dialogue_id = ?",
        (goal_state_id, meta, dialogue["dialogue_id"]),
    )


def finalize_dialogue_for_current_task(conn, *, conversation, task_key=None, task_index=None, event=None):
    if not (g.user and g.session):
        return None
    dialogue = _get_active_dialogue(conn, conversation_id=conversation["id"], task_key=task_key or conversation["example_key"])
    if not dialogue:
        return None
    path = _json_loads_safe(dialogue["full_path"], [])
    if not isinstance(path, list):
        path = []
    current_state = _normalize_path_state(conversation["current_state"])
    if current_state and (not path or path[-1] != current_state):
        path.append(current_state)
    is_success = int(is_stop_state_value(conversation["current_state"]))
    meta = _merge_json_text(
        dialogue["meta"],
        {
            "completed_event": event,
            "completed_task_index": task_index,
            "completed_state": conversation["current_state"],
            "success_rule": "reach_any_stop_node",
        },
    )
    conn.execute(
        """
        UPDATE dialogue
        SET ended_at = COALESCE(ended_at, ?),
            flowchart_alias = COALESCE(flowchart_alias, ?),
            task_key = COALESCE(task_key, ?),
            full_path = ?,
            path_node_count = ?,
            is_success = ?,
            meta = ?
        WHERE dialogue_id = ?
        """,
        (
            utc_now_iso(),
            conversation["example_key"],
            task_key or conversation["example_key"],
            _json_dumps_safe(path),
            len(path),
            is_success,
            meta,
            dialogue["dialogue_id"],
        ),
    )
    cur = conn.execute("SELECT * FROM dialogue WHERE dialogue_id = ?", (dialogue["dialogue_id"],))
    return cur.fetchone()


def finalize_experiment_session_row(conn, details=None):
    if not (g.user and g.session):
        return
    ensure_experiment_session_row(conn, {"client_meta": details or {}})
    rows = conn.execute(
        "SELECT task_key, is_success, started_at, dialogue_id FROM dialogue WHERE session_id = ? ORDER BY started_at DESC, dialogue_id DESC",
        (g.session["id"],),
    ).fetchall()
    latest_by_task = {}
    for row in rows:
        task_key = row["task_key"] or f"__dialogue__:{row['dialogue_id']}"
        if task_key not in latest_by_task:
            latest_by_task[task_key] = row
    selected_rows = list(latest_by_task.values())
    total = len(selected_rows)
    success_count = sum(int(r["is_success"]) for r in selected_rows)
    is_success = int(total > 0 and success_count == total)
    existing = conn.execute(
        'SELECT client_meta FROM "session" WHERE session_id = ?',
        (g.session["id"],),
    ).fetchone()
    meta = _merge_json_text(
        existing["client_meta"] if existing else None,
        {
            "dialogue_count": total,
            "successful_dialogue_count": success_count,
            "dialogue_count_raw": len(rows),
            "success_rule": "all_tasks_reach_any_stop_node",
        },
    )
    conn.execute(
        """
        UPDATE "session"
        SET ended_at = COALESCE(ended_at, ?),
            status = 'completed',
            is_success = ?,
            client_meta = ?
        WHERE session_id = ?
        """,
        (utc_now_iso(), is_success, meta, g.session["id"]),
    )


# ==================== 请求钩子 ====================

@app.before_request
def before_request():
    g.user, g.session = get_auth_context()

    rl = rate_limit_guard()
    if rl:
        return rl

    sec = security_guards()
    if sec:
        return sec


# ==================== Flask路由 ====================

@app.route('/')
def index():
    """主页"""
    return render_template(
        'index.html',
        user_study_mode=False,
        ui_texts_bundle=get_ui_texts_bundle(),
        default_ui_lang=get_ui_lang(),
    )


@app.route('/user-study')
def user_study():
    """用户调研页面"""
    return render_template(
        'index.html',
        user_study_mode=True,
        ui_texts_bundle=get_ui_texts_bundle(),
        default_ui_lang=get_ui_lang(),
    )


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    user = authenticate_user(username, password)
    if not user:
        log_audit(None, "login", 401, decision="failed")
        return jsonify({"success": False, "error": "invalid credentials"}), 401

    token, csrf_token, session_id = create_session(
        user["id"], request.remote_addr, request.headers.get("User-Agent")
    )

    allowed_examples = get_allowed_example_keys(user["id"])
    if not allowed_examples:
        log_audit(user, "login", 403, decision="no_example_access")
        return jsonify({"success": False, "error": "no example access"}), 403

    conversation = get_or_create_conversation(user["id"], pick_default_example(allowed_examples))
    conversation = ensure_start_state(conversation)
    set_session_conversation(session_id, conversation["id"])

    log_audit(user, "login", 200, decision="success")
    return jsonify({
        "success": True,
        "token": token,
        "csrf_token": csrf_token,
        "user": {"id": user["id"], "username": user["username"]},
        "allowed_examples": allowed_examples,
        "conversation_id": conversation["id"],
    })


@app.route('/api/logout', methods=['POST'])
def logout():
    resp = require_permission("chat", "logout")
    if resp:
        return resp

    if g.session:
        revoke_session(g.session["id"])
    log_audit(g.user, "logout", 200, decision="success")
    return jsonify({"success": True})


@app.route('/api/users', methods=['POST'])
def create_user():
    resp = require_permission("create_user", "create_user")
    if resp:
        return resp

    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    role = data.get("role", "guest").strip()
    attrs = data.get("attrs", {}) or {}

    if not username or not password:
        return jsonify({"success": False, "error": "username and password required"}), 400

    if role not in ["guest", "admin"]:
        return jsonify({"success": False, "error": "invalid role"}), 400

    with get_db() as conn:
        cur = conn.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cur.fetchone():
            return jsonify({"success": False, "error": "user exists"}), 409
        conn.execute(
            "INSERT INTO users (username, password_hash, is_active, attrs_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (username, generate_password_hash(password), 1, json.dumps(attrs, ensure_ascii=False), utc_now_iso()),
        )

    from acl import assign_role_to_user
    assign_role_to_user(username, role)
    log_audit(g.user, "create_user", 201, decision="success", message=f"{username}:{role}")
    return jsonify({"success": True})


@app.route('/api/me')
def me():
    resp = require_permission("chat", "me")
    if resp:
        return resp

    log_audit(g.user, "me", 200, decision="success")
    return jsonify({
        "success": True,
        "user": {"id": g.user["id"], "username": g.user["username"]}
    })


@app.route('/graph')
def graph():
    """获取流程图数据"""
    resp = require_permission("view_examples", "graph")
    if resp:
        return resp

    lang = get_ui_lang()
    current_conv = get_current_conversation(g.session["id"]) if g.session else None
    example_key = current_conv["example_key"] if current_conv else DEFAULT_EXAMPLE_KEY
    allowed = get_allowed_example_keys(g.user["id"])
    if example_key not in allowed and not can_access_study_examples_without_acl():
        example_key = pick_default_example(allowed)

    data = run(get_example(example_key, lang=lang))
    log_audit(g.user, "graph", 200, decision="success", resource_key=example_key)
    return jsonify(data)


@app.route('/api/workflow_data')
def get_workflow_data():
    resp = require_permission("view_examples", "load_workflow_data")
    if resp:
        return resp

    reset_flag = _is_truthy_param(request.args.get("reset"))

    lang = get_ui_lang()
    allowed = get_allowed_example_keys(g.user["id"])
    examples = {k: v for k, v in get_all_examples(lang=lang).items() if k in allowed}
    if can_access_study_examples_without_acl():
        examples = get_all_examples(lang=lang)
    visible_keys = set(VISIBLE_UML_EXAMPLE_KEYS)
    examples = {k: v for k, v in examples.items() if k in visible_keys}

    if not examples:
        log_audit(g.user, "load_workflow_data", 403, decision="no_example_access")
        return jsonify({"success": False, "error": "no example access"}), 403

    current_conv = get_current_conversation(g.session["id"]) if g.session else None
    current_conv = ensure_start_state(current_conv)
    if not current_conv or current_conv["example_key"] not in examples:
        current_example = pick_default_example(list(examples.keys()))
        current_conv = get_or_create_conversation(g.user["id"], current_example)
        current_conv = ensure_start_state(current_conv)
        if g.session:
            set_session_conversation(g.session["id"], current_conv["id"])
    else:
        current_example = current_conv["example_key"]

    if reset_flag and current_conv:
        current_conv = reset_conversation_state(
            current_conv,
            trigger="page_load",
            reason="page_reload",
            clear_messages=False,
            action_type="reset",
        )

    plantuml = get_example(current_example, lang=lang)
    log_audit(g.user, "load_workflow_data", 200, decision="success", resource_key=current_example)

    return jsonify({
        "success": True,
        "examples": examples,
        "plantuml": plantuml,
        "nodes": parse_plantuml_to_nodes(plantuml),
        "current_state": localize_state_for_ui(current_example, current_conv["current_state"], lang),
        "conversation_id": current_conv["id"],
        "current_example_key": current_example,
    })


@app.route('/api/user_test_tasks')
def user_test_tasks():
    resp = require_permission("view_examples", "user_test_tasks")
    if resp:
        return resp

    lang = get_ui_lang()
    # Keep resource metadata in sync when new task examples are added.
    sync_resources()
    allowed = set(get_allowed_example_keys(g.user["id"]))
    if can_access_study_examples_without_acl():
        allowed = set(get_all_examples(lang=lang).keys())
    tasks = []
    visible_index = 0
    for key, info in get_user_test_tasks(lang=lang):
        if key in allowed:
            visible_index += 1
            base_title = info.get("name", key)
            title = f"任务 {visible_index} - {base_title}" if lang == "zh" else f"Task {visible_index} - {base_title}"
            tasks.append({
                "key": key,
                "title": title,
                "description": info.get("description", ""),
                "plantuml": info.get("plantuml", ""),
            })

    if not tasks:
        log_audit(g.user, "user_test_tasks", 403, decision="no_example_access")
        return jsonify({"success": False, "error": "no example access"}), 403

    log_audit(g.user, "user_test_tasks", 200, decision="success", message=",".join([t["key"] for t in tasks]))
    return jsonify({"success": True, "tasks": tasks})


@app.route('/api/user_test_tasks/<task_key>/start', methods=['POST'])
def start_user_test_task(task_key):
    data = request.get_json(silent=True) or {}
    lang = get_ui_lang(data)
    resp = require_permission("chat", "start_user_test_task", resource_key=task_key)
    if resp:
        return resp

    all_examples = get_all_examples(lang=lang)
    if task_key not in all_examples:
        log_audit(g.user, "start_user_test_task", 404, decision="not_found", resource_key=task_key)
        return jsonify({"success": False, "error": "invalid task key"}), 404

    sync_resources()
    allowed = set(get_allowed_example_keys(g.user["id"]))
    if can_access_study_examples_without_acl():
        allowed = set(all_examples.keys())
    if task_key not in allowed:
        log_audit(g.user, "start_user_test_task", 403, decision="forbidden", resource_key=task_key)
        return jsonify({"success": False, "error": "forbidden"}), 403

    conversation = get_or_create_conversation(g.user["id"], task_key)
    with get_db() as conn:
        ensure_experiment_session_row(conn, data)
        conn.execute(
            "UPDATE conversations SET current_state = ?, updated_at = ? WHERE id = ?",
            ("<start>", utc_now_iso(), conversation["id"]),
        )
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation["id"],))
        cur = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation["id"],))
        conversation = cur.fetchone()
        try:
            _create_study_dialogue(
                conn,
                conversation,
                task_key,
                {
                    "task_index": data.get("task_index"),
                    "total_tasks": data.get("total_tasks"),
                    "task_title": all_examples.get(task_key, {}).get("name", task_key),
                    "client_tz": data.get("client_tz"),
                    "client_meta": data.get("client_meta"),
                },
            )
        except Exception as e:
            print(f"Study analytics: start task logging skipped: {e}", flush=True)

    if g.session:
        set_session_conversation(g.session["id"], conversation["id"])

    info = all_examples.get(task_key, {})
    plantuml = info.get("plantuml", get_example(task_key, lang=lang))

    log_audit(g.user, "start_user_test_task", 200, decision="success", resource_key=task_key)
    return jsonify({
        "success": True,
        "task": {
            "key": task_key,
            "title": info.get("name", task_key),
            "description": info.get("description", ""),
        },
        "plantuml": plantuml,
        "conversation_id": conversation["id"],
        "current_state": conversation["current_state"],
        "nodes": parse_plantuml_to_nodes(plantuml),
    })


@app.route('/api/graph/jump', methods=['POST'])
def graph_jump():
    resp = require_permission("chat", "graph_jump")
    if resp:
        return resp

    data = request.get_json(silent=True) or {}
    lang = get_ui_lang(data)
    conversation_id = data.get("conversation_id")
    node_id = data.get("node_id")
    target_label = (data.get("target_label") or "").strip()
    to_state = (data.get("to_state") or "").strip()
    action_type = (data.get("action_type") or "jump").strip()
    trigger = (data.get("trigger") or "graph_click").strip()
    from_state = (data.get("from_state") or "").strip()

    if not to_state:
        to_state = target_label or (str(node_id).strip() if node_id is not None else "")

    if not to_state:
        return jsonify({"success": False, "error": "target state required"}), 400

    if conversation_id:
        with get_db() as conn:
            cur = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
            conversation = cur.fetchone()
    else:
        conversation = get_current_conversation(g.session["id"]) if g.session else None

    if not can_access_conversation(g.user["id"], conversation):
        log_audit(g.user, "graph_jump", 403, decision="forbidden")
        return jsonify({"success": False, "error": "forbidden"}), 403

    example_key = conversation["example_key"]
    plantuml = get_example(example_key, lang=lang)
    to_state = normalize_state_name(to_state, plantuml)
    to_state_for_db = to_chinese_state_for_storage(example_key, to_state, lang)

    if not from_state:
        from_state_for_db = conversation["current_state"]
    else:
        from_state = normalize_state_name(from_state, plantuml)
        from_state_for_db = to_chinese_state_for_storage(example_key, from_state, lang)

    with get_db() as conn:
        conn.execute(
            "UPDATE conversations SET current_state = ?, updated_at = ? WHERE id = ?",
            (to_state_for_db, utc_now_iso(), conversation["id"]),
        )
        log_transition_event(
            conversation_id=conversation["id"],
            user_id=conversation["user_id"],
            example_key=example_key,
            action_type=action_type or "jump",
            from_state=from_state_for_db,
            to_state=to_state_for_db,
            trigger=trigger,
            payload={
                "node_id": node_id,
                "target_label": target_label,
            },
            conn=conn,
        )

    to_state_ui = localize_state_for_ui(example_key, to_state_for_db, lang)
    state_to_node_map, plantuml_nodes = build_state_mapping(plantuml)
    current_node_index = state_to_node_map.get(to_state_ui, 0)
    completed_nodes = list(range(0, current_node_index))
    workflow_state = {
        "current_node": current_node_index + 1,
        "completed_nodes": completed_nodes,
        "nodes": [{"id": i + 1, "name": name, "type": "activity"}
                  for i, name in enumerate(["<start>"] + plantuml_nodes + ["<end>"])]
    }

    log_audit(g.user, "graph_jump", 200, decision="success", resource_key=example_key)
    return jsonify({
        "success": True,
        "current_state": to_state_ui,
        "workflow_state": workflow_state,
    })


@app.route('/api/task_event', methods=['POST'])
def task_event():
    resp = require_permission("chat", "task_event")
    if resp:
        return resp

    data = request.get_json(silent=True) or {}
    event = (data.get("event") or "").strip()
    task_key = (data.get("task_key") or "").strip()
    task_index = data.get("task_index")
    details = data.get("details")

    if not event:
        return jsonify({"success": False, "error": "event required"}), 400

    payload = {
        "event": event,
        "task_key": task_key,
        "task_index": task_index,
        "details": details,
    }

    if g.session and is_study_mode_request():
        try:
            with get_db() as conn:
                ensure_experiment_session_row(conn, {"client_meta": details if isinstance(details, dict) else None})
                conversation = get_current_conversation(g.session["id"])
                if conversation and can_access_conversation(g.user["id"], conversation):
                    if event == "goal_selected":
                        update_dialogue_goal_selection(
                            conn,
                            conversation=conversation,
                            task_key=task_key or conversation["example_key"],
                            details=details if isinstance(details, dict) else {},
                            task_index=task_index,
                        )
                    elif event == "task_completed":
                        finalize_dialogue_for_current_task(
                            conn,
                            conversation=conversation,
                            task_key=task_key or conversation["example_key"],
                            task_index=task_index,
                            event=event,
                        )
                    elif event == "study_completed":
                        finalize_experiment_session_row(
                            conn,
                            details={"study_completed_event": True, "task_index": task_index},
                        )
        except Exception as e:
            print(f"Study analytics: task_event logging skipped: {e}", flush=True)

    log_audit(g.user, "task_event", 200, decision="success", resource_key=task_key, message=json.dumps(payload, ensure_ascii=False))
    return jsonify({"success": True})


@app.route('/api/switch_example/<example_key>')
def switch_example(example_key):
    lang = get_ui_lang()
    resp = require_permission("switch_example", "switch_example", resource_key=example_key)
    if resp:
        return resp

    allowed = get_allowed_example_keys(g.user["id"])
    if example_key not in allowed and not can_access_study_examples_without_acl():
        log_audit(g.user, "switch_example", 403, decision="forbidden", resource_key=example_key)
        return jsonify({"success": False, "error": "forbidden"}), 403

    example = get_example(example_key, lang=lang)
    if not example:
        log_audit(g.user, "switch_example", 404, decision="not_found", resource_key=example_key)
        return jsonify({"success": False, "error": "invalid example key"}), 404

    conversation = get_or_create_conversation(g.user["id"], example_key)
    if g.session:
        set_session_conversation(g.session["id"], conversation["id"])

    log_transition_event(
        conversation_id=conversation["id"],
        user_id=conversation["user_id"],
        example_key=example_key,
        action_type="switch_example",
        from_state=conversation["current_state"],
        to_state=conversation["current_state"],
        trigger="task_select",
        payload={"example_key": example_key},
    )

    log_audit(g.user, "switch_example", 200, decision="success", resource_key=example_key)
    return jsonify({
        "success": True,
        "plantuml": example,
        "nodes": parse_plantuml_to_nodes(example),
        "current_state": localize_state_for_ui(example_key, conversation["current_state"], lang),
        "conversation_id": conversation["id"],
        "current_example_key": example_key,
    })


@app.route('/api/model_info')
def get_model_info():
    resp = require_permission("model_info", "model_info")
    if resp:
        return resp

    log_audit(g.user, "model_info", 200, decision="success")
    return jsonify({
        "success": True,
        "model_info": get_current_model_info()
    })


@app.route('/api/switch_model/<model_type>')
def api_switch_model(model_type):
    resp = require_permission("switch_model", "switch_model")
    if resp:
        return resp

    if model_type not in ["dummy", "real"]:
        log_audit(g.user, "switch_model", 400, decision="invalid")
        return jsonify({"success": False, "error": "invalid model type"}), 400

    try:
        model = switch_model(model_type)
        log_audit(g.user, "switch_model", 200, decision="success")
        return jsonify({
            "success": True,
            "model_info": model.get_model_info()
        })
    except Exception as e:
        log_audit(g.user, "switch_model", 500, decision="error", message=str(e))
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    resp = require_permission("chat", "chat")
    if resp:
        return resp

    data = request.get_json(silent=True) or {}
    lang = get_ui_lang(data)
    user_message = data.get("message", "").strip()
    conversation_id = data.get("conversation_id")

    if not user_message:
        return jsonify({"success": False, "error": "empty message"}), 400

    # 确保模型已加载
    if not ensure_model_loaded():
        return jsonify({
            "success": False,
            "error": "模型未加载。请设置 FLOWGUARD_MODEL_TYPE=dummy 或配置真实模型路径。"
        }), 503

    if conversation_id:
        with get_db() as conn:
            cur = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
            conversation = cur.fetchone()
    else:
        conversation = get_current_conversation(g.session["id"]) if g.session else None

    if not can_access_conversation(g.user["id"], conversation):
        log_audit(g.user, "chat", 403, decision="forbidden")
        return jsonify({"success": False, "error": "forbidden"}), 403

    example_key = conversation["example_key"]
    allowed = get_allowed_example_keys(g.user["id"])
    if example_key not in allowed and not can_access_study_examples_without_acl():
        log_audit(g.user, "chat", 403, decision="forbidden", resource_key=example_key)
        return jsonify({"success": False, "error": "forbidden"}), 403

    plantuml = get_example(example_key, lang=lang)

    # 构建对话内容
    current_state = conversation["current_state"]
    current_state_ui = localize_state_for_ui(example_key, current_state, lang)
    next_state_candidates_ui = get_next_state_candidates(current_state_ui, plantuml)
    allowed_next_states_text = ", ".join(next_state_candidates_ui) if next_state_candidates_ui else "(none)"

    conversation_prompt = [
        {
            "role": "system",
            "content": config.get_system_prompt(plantuml, lang == "en")
        },
        {
            "role": "user",
            "content": config.get_user_prompt(
                current_state_ui,
                user_message,
                lang == "en",
                allowed_next_states=allowed_next_states_text,
            )
        }
    ]

    turn_ts_start = utc_now_iso()
    storage_lang = normalize_lang(lang)
    zh_plantuml = get_example(example_key, lang="zh")
    current_state_for_db = to_chinese_state_for_storage(example_key, current_state, storage_lang)
    print(f"Chat: 当前状态 = {current_state}", flush=True)
    print(f"Chat: 用户输入 = {user_message}", flush=True)

    # 使用模型生成回复
    try:
        model = get_model()
        next_state, response = model.generate(conversation_prompt, plantuml=plantuml)
    except Exception as e:
        print(f"Chat: 生成失败: {e}", flush=True)
        log_audit(g.user, "chat", 500, decision="error", message=str(e))
        return jsonify({"success": False, "error": f"生成失败: {str(e)}"}), 500
    next_state = normalize_state_name(next_state, plantuml)
    next_state = resolve_next_state_from_candidates(next_state, next_state_candidates_ui, current_state_ui)
    next_state_for_db = to_chinese_state_for_storage(example_key, next_state, storage_lang)
    next_state_ui = localize_state_for_ui(example_key, next_state_for_db, lang)
    turn_ts_end = utc_now_iso()

    # 更新状态
    with get_db() as conn:
        conn.execute(
            "UPDATE conversations SET current_state = ?, updated_at = ? WHERE id = ?",
            (next_state_for_db, utc_now_iso(), conversation["id"]),
        )
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, message, timestamp) VALUES (?, ?, ?, ?)",
            (conversation["id"], "user", user_message, time.time()),
        )
        user_message_id = cur.lastrowid
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, message, timestamp) VALUES (?, ?, ?, ?)",
            (conversation["id"], "assistant", response, time.time()),
        )
        assistant_message_id = cur.lastrowid
        conn.execute(
            """
            INSERT INTO flowchart (alias, content)
            VALUES (?, ?)
            ON CONFLICT(alias) DO UPDATE SET content = excluded.content
            """,
            (example_key, zh_plantuml),
        )
        conn.execute(
            "INSERT INTO five_tuples (flowchart, current_state, next_state, user_input, robot_output, lang) VALUES (?, ?, ?, ?, ?, ?)",
            (example_key, current_state_for_db, next_state_for_db, user_message, response, storage_lang),
        )
        log_transition_event(
            conversation_id=conversation["id"],
            user_id=conversation["user_id"],
            example_key=example_key,
            action_type="model",
            from_state=current_state_for_db,
            to_state=next_state_for_db,
            trigger="text",
            message_id=user_message_id,
            payload={
                "user_input": user_message,
                "robot_output": response,
                "assistant_message_id": assistant_message_id,
            },
            conn=conn,
        )
        try:
            append_dialogue_path_and_turn(
                conn,
                conversation=conversation,
                current_state=current_state_for_db,
                next_state=next_state_for_db,
                user_message=user_message,
                system_reply=response,
                ts_start=turn_ts_start,
                ts_end=turn_ts_end,
            )
        except Exception as e:
            print(f"Study analytics: chat turn logging skipped: {e}", flush=True)

    # 构建工作流状态数据
    state_to_node_map, plantuml_nodes = build_state_mapping(plantuml)
    current_node_index = state_to_node_map.get(next_state_ui, 0)
    completed_nodes = list(range(0, current_node_index))

    workflow_state = {
        "current_node": current_node_index + 1,
        "completed_nodes": completed_nodes,
        "nodes": [{"id": i + 1, "name": name, "type": "activity"}
                  for i, name in enumerate(["<start>"] + plantuml_nodes + ["<end>"])]
    }

    log_audit(g.user, "chat", 200, decision="success", resource_key=example_key)
    return jsonify({
        "success": True,
        "response": response,
        "current_state": next_state_ui,
        "workflow_state": workflow_state,
        "conversation_id": conversation["id"],
    })


@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    resp = require_permission("chat", "reset")
    if resp:
        return resp

    data = request.get_json(silent=True) or {}
    conversation_id = data.get("conversation_id")

    if conversation_id:
        with get_db() as conn:
            cur = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
            conversation = cur.fetchone()
    else:
        conversation = get_current_conversation(g.session["id"]) if g.session else None

    if not can_access_conversation(g.user["id"], conversation):
        log_audit(g.user, "reset", 403, decision="forbidden")
        return jsonify({"success": False, "error": "forbidden"}), 403

    reset_conversation_state(
        conversation,
        trigger="button",
        reason="user_reset",
        clear_messages=True,
        action_type="reset",
    )

    log_audit(g.user, "reset", 200, decision="success", resource_key=conversation["example_key"])
    return jsonify({"success": True, "status": "reset", "conversation_id": conversation["id"]})


@app.route('/api/render_uml', methods=['POST'])
def render_uml():
    resp = require_permission("chat", "render_uml")
    if resp:
        return resp

    try:
        data = request.get_json(silent=True) or {}
        plantuml = data.get('plantuml', '')
        if not plantuml:
            return jsonify({"success": False, "error": "empty plantuml"}), 400

        req = urllib.request.Request(
            url='https://kroki.io/plantuml/svg',
            data=plantuml.encode('utf-8'),
            headers={'Content-Type': 'text/plain'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            svg = resp.read().decode('utf-8', errors='ignore')

        log_audit(g.user, "render_uml", 200, decision="success")
        return jsonify({"success": True, "svg": svg})
    except urllib.error.HTTPError as e:
        log_audit(g.user, "render_uml", 502, decision="error", message=str(e))
        return jsonify({"success": False, "error": f"kroki http {e.code}"}), 502
    except urllib.error.URLError as e:
        log_audit(g.user, "render_uml", 502, decision="error", message=str(e))
        return jsonify({"success": False, "error": f"kroki url error: {e.reason}"}), 502
    except Exception as e:
        log_audit(g.user, "render_uml", 500, decision="error", message=str(e))
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/user_operations')
def get_user_operations_api():
    resp = require_permission("view_audit", "audit")
    if resp:
        return resp

    limit = request.args.get("limit", default=100, type=int)
    limit = max(1, min(limit, 500))

    with get_db() as conn:
        cur = conn.execute(
            "SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        records = [dict(row) for row in cur.fetchall()]

    log_audit(g.user, "audit", 200, decision="success")
    return jsonify({
        "success": True,
        "count": len(records),
        "operations": records
    })


# ==================== 主程序入口 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("FlowGuard 启动中...")
    print(f"模型类型: {config.MODEL_TYPE}")
    print(f"默认示例: {DEFAULT_EXAMPLE_KEY}")
    print("=" * 60)

    app.run(
        debug=config.FLASK_DEBUG,
        host=config.FLASK_HOST,
        port=config.FLASK_PORT
    )
    lang = get_ui_lang()
