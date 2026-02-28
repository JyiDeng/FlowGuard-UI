# -*- coding: utf-8 -*-
"""
RBAC/ABAC and resource ACL helpers.
"""
import json

from db import get_db, utc_now_iso

PERMISSIONS = [
    "view_examples",
    "switch_example",
    "chat",
    "model_info",
    "switch_model",
    "view_audit",
    "create_user",
]

ROLE_PERMISSIONS = {
    "admin": PERMISSIONS,
    "guest": ["view_examples", "switch_example", "chat", "model_info"],
}


def ensure_roles_permissions():
    with get_db() as conn:
        cur = conn.cursor()
        for role in ROLE_PERMISSIONS.keys():
            cur.execute("INSERT OR IGNORE INTO roles (name) VALUES (?)", (role,))

        for perm in PERMISSIONS:
            cur.execute("INSERT OR IGNORE INTO permissions (name) VALUES (?)", (perm,))

        for role, perms in ROLE_PERMISSIONS.items():
            cur.execute("SELECT id FROM roles WHERE name = ?", (role,))
            role_id = cur.fetchone()["id"]
            for perm in perms:
                cur.execute("SELECT id FROM permissions WHERE name = ?", (perm,))
                perm_id = cur.fetchone()["id"]
                cur.execute(
                    "INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES (?, ?)",
                    (role_id, perm_id),
                )


def assign_role_to_user(username, role_name):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        user = cur.fetchone()
        if not user:
            return
        cur.execute("SELECT id FROM roles WHERE name = ?", (role_name,))
        role = cur.fetchone()
        if not role:
            return
        cur.execute(
            "INSERT OR IGNORE INTO user_roles (user_id, role_id) VALUES (?, ?)",
            (user["id"], role["id"]),
        )


def get_user_roles(user_id):
    with get_db() as conn:
        cur = conn.execute(
            "SELECT r.name FROM roles r JOIN user_roles ur ON ur.role_id = r.id WHERE ur.user_id = ?",
            (user_id,),
        )
        return [row["name"] for row in cur.fetchall()]


def user_has_permission(user_id, permission):
    with get_db() as conn:
        cur = conn.execute(
            "SELECT 1 FROM permissions p\n"
            "JOIN role_permissions rp ON rp.permission_id = p.id\n"
            "JOIN user_roles ur ON ur.role_id = rp.role_id\n"
            "WHERE ur.user_id = ? AND p.name = ?",
            (user_id, permission),
        )
        return cur.fetchone() is not None


def upsert_resource(resource_key, resource_type, name, policy):
    policy_json = json.dumps(policy, ensure_ascii=False)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM resources WHERE resource_key = ?", (resource_key,))
        existing = cur.fetchone()
        if existing:
            cur.execute(
                "UPDATE resources SET name = ?, policy_json = ? WHERE resource_key = ?",
                (name, policy_json, resource_key),
            )
        else:
            cur.execute(
                "INSERT INTO resources (resource_key, resource_type, name, policy_json) VALUES (?, ?, ?, ?)",
                (resource_key, resource_type, name, policy_json),
            )


def ensure_resource_acls_for_defaults():
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM roles WHERE name = 'admin'")
        admin_role = cur.fetchone()
        if not admin_role:
            return

        cur.execute("SELECT id FROM resources")
        for row in cur.fetchall():
            res_id = row["id"]
            cur.execute(
                "INSERT OR IGNORE INTO resource_acl (resource_id, role_id, allow) VALUES (?, ?, 1)",
                (res_id, admin_role["id"]),
            )


def get_resource(resource_key):
    with get_db() as conn:
        cur = conn.execute("SELECT * FROM resources WHERE resource_key = ?", (resource_key,))
        return cur.fetchone()


def get_allowed_example_keys(user_id):
    roles = set(get_user_roles(user_id))
    with get_db() as conn:
        cur = conn.execute("SELECT * FROM resources WHERE resource_type = 'example'")
        rows = cur.fetchall()

    allowed = []
    for res in rows:
        if _is_resource_allowed(user_id, roles, res):
            allowed.append(res["resource_key"])
    return allowed


def _is_resource_allowed(user_id, roles, resource_row):
    policy = json.loads(resource_row["policy_json"] or "{}")
    policy_roles = set(policy.get("roles", []))
    policy_attrs = policy.get("attrs", {})

    user_attrs = {}
    with get_db() as conn:
        cur = conn.execute("SELECT attrs_json FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        if row:
            try:
                user_attrs = json.loads(row["attrs_json"] or "{}")
            except Exception:
                user_attrs = {}

    role_match = not policy_roles or bool(roles & policy_roles)
    attr_match = True
    for key, value in policy_attrs.items():
        if user_attrs.get(key) != value:
            attr_match = False
            break

    if role_match and attr_match:
        return True

    with get_db() as conn:
        cur = conn.execute(
            "SELECT ra.allow FROM resource_acl ra\n"
            "LEFT JOIN user_roles ur ON ur.role_id = ra.role_id\n"
            "WHERE ra.resource_id = ? AND (ra.user_id = ? OR ur.user_id = ?)",
            (resource_row["id"], user_id, user_id),
        )
        for row in cur.fetchall():
            if row["allow"]:
                return True

    return False
