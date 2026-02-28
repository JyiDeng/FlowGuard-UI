# FlowGuard UI

FlowGuard UI is the web UI and service implementation for FlowGuard, a process-driven customer-service dialogue system that combines LLMs with explicit workflow diagrams specified in PlantUML.

Accurate state access and traceability are crucial in real-world customer-service automation, yet end-to-end LLM agents often suffer from uncontrollable state transitions, untraceable processes, and high fine-tuning cost. FlowGuard maps each response to a workflow state, supports node rollback for error recovery, and provides a web platform for workflow visualization, state tracking, and interactive testing. Experiments and user studies show strong workflow compliance, usability, response efficiency, and user experience, suggesting a practical path toward controllable, auditable, and cost-efficient customer-service Q&A.

## Requirements

- Python 3.10+
- pip

## Install

```bash
pip install -r requirements.txt
```

## Run (real model)

A real model will traverse to the correct UML nodes according to the given user input.

```bash
export FLOWGUARD_MODEL_TYPE=real
export FLOWGUARD_MODEL_PATH=/path/to/your/model
export FLOWGUARD_DEVICE=cuda:0  # or cpu
python main.py
```

## Run (dummy model for pure UI test)

A dummy model will traverse all the UML nodes in sequence, whatever the given user input.

```bash
export FLOWGUARD_MODEL_TYPE=dummy
python main.py
```

## Visit UI

For development mode:

Open guest mode: http://localhost:FLASK_PORT

Open user test mode: http://localhost:FLASK_PORT/user-test


## Project Structure And File Relations

- `main.py` is the main Flask app. It routes HTTP APIs and UI pages, loads workflows, dispatches model calls, and records audit logs.
- `examples.py` defines workflow examples in PlantUML and provides lookup utilities. `main.py` reads these workflows for visualization and state transitions.
- `models/` contains model adapters. `main.py` calls `models.get_model()` and `models.switch_model()` to choose dummy or real models.
- `parser.py` parses PlantUML and derives workflow states and transitions for runtime state management.
- `db.py` defines the SQLite schema and database utilities. `main.py` initializes and uses it for sessions, messages, audits, and analytics tables.
- `acl.py`, `auth.py`, `security.py`, `rate_limit.py`, `audit.py` provide access control, authentication, request validation, rate limiting, and audit logging.
- `templates/` contains the web UI pages used by `main.py` (workflow visualization, chat, user testing).
- `ui_texts.py` provides UI text bundles for multi-language rendering.
- `config.py` stores configuration defaults and validation logic.

## Add Your Own PlantUML Workflow

1. Open `examples.py`.
2. Add a new PlantUML string, for example `EXAMPLE_MY_FLOW = """@startuml ... @enduml"""`.
3. Add an entry to `AVAILABLE_EXAMPLES` with a unique key:
   - `name`
   - `plantuml` set to your new string
   - `description`
   - `policy` (optional, controls who can access it)
4. Add an i18n entry in `_EXAMPLE_I18N` for the same key, at least one of `en` or `zh`.
5. If you want it visible in the UI selector, add the key to `VISIBLE_UML_EXAMPLE_KEYS`.

Restart the server. The new workflow will appear in the UI and can be loaded by API.

## Database Usage

FlowGuard uses SQLite and creates the database automatically on startup.

- Default path: `data/app.db`
- Override path: set `FLOWGUARD_DB_PATH` before starting the app.

Schema and migrations are defined in `db.py` and are applied in `init_db()` when `main.py` starts.

Basic usage:

```bash
# Use default path
python main.py

# Use a custom path
export FLOWGUARD_DB_PATH=/path/to/app.db
python main.py
```

Inspect the database:

```bash
sqlite3 data/app.db ".tables"
sqlite3 data/app.db "SELECT * FROM users LIMIT 5;"
```

## Batch Generate Test Users (User Credentials)

You can batch create users through the admin API. This produces a list of credentials you can share with testers.

1. Log in as admin to get a token:

```bash
curl -s -X POST http://localhost:FLASK_PORT/api/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'
```

2. Create users in batch (example Bash loop):

```bash
TOKEN="<paste-token>"

for i in $(seq -w 1 20); do
  username="test_user_${i}"
  password="test_pass_${i}"
  curl -s -X POST http://localhost:FLASK_PORT/api/users \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${TOKEN}" \
    -d "{\"username\":\"${username}\",\"password\":\"${password}\",\"role\":\"guest\",\"attrs\":{\"study\":\"user_test\"}}"
  echo "${username} / ${password}"
done
```

Notes:

- Only `admin` can call `/api/users`.
- Store the printed `username / password` lines as your User Credentials list.
- `attrs.study` is optional and can be removed if not needed.

## Default Accounts

- guest / guest123
- admin / admin123

## API Auth

All protected endpoints require either:

- `Authorization: Bearer <token>` from `/api/login`
- or `X-API-Key: <key>` (default admin key: `admin-api-key`)

State-changing calls require:

- `X-CSRF-Token` (for session token)
- `X-Request-Nonce`
- `X-Request-Timestamp` (epoch seconds)

## Environment Variables

- `FLOWGUARD_MODEL_TYPE`: `dummy` or `real`
- `FLOWGUARD_MODEL_PATH`: path to model (real only)
- `FLOWGUARD_DEVICE`: device string (e.g. `cuda:0`, `cpu`)
- `FLOWGUARD_ENV`: `development` or `production`
- `FLOWGUARD_SECRET_KEY`: required in production
- `FLOWGUARD_RATE_LIMIT`: per-minute limit

## API

- `POST /api/login`
- `POST /api/logout`
- `GET /api/me`
- `GET /api/model_info`
- `GET /api/switch_model/<type>`
- `GET /api/workflow_data`
- `GET /api/switch_example/<key>`
- `POST /api/chat`
- `POST /api/reset`
- `POST /api/render_uml`
- `GET /api/user_operations`
