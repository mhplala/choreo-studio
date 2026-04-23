"""SQLite persistence for Choreo Studio.

Schema:
  projects(id, name, script, created_at, updated_at)
  elements(id, project_id, kind, name, tos_key, source, prompt, created_at)
  shots(id, project_id, order_idx, prompt, element_ids_json, duration_sec,
        video_tos_key, status, task_id, seed, created_at, updated_at)
  tracks(id, project_id, kind, tos_key, start_sec, volume, created_at)

All timestamps are unix seconds (REAL). No user table — single-tenant behind Basic Auth.
"""
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

_LOCK = threading.Lock()
_DB_PATH: Path | None = None


def init_db(db_path: Path):
    global _DB_PATH
    _DB_PATH = db_path
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                script      TEXT DEFAULT '',
                settings    TEXT NOT NULL DEFAULT '{}',
                bible       TEXT NOT NULL DEFAULT '{}',
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS elements (
                id          TEXT PRIMARY KEY,
                project_id  TEXT NOT NULL,
                kind        TEXT NOT NULL,        -- character | style | prop
                name        TEXT NOT NULL,
                tos_key     TEXT NOT NULL,
                source      TEXT NOT NULL,        -- upload | generated
                prompt      TEXT DEFAULT '',
                created_at  REAL NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_elements_project ON elements(project_id);

            CREATE TABLE IF NOT EXISTS shots (
                id                TEXT PRIMARY KEY,
                project_id        TEXT NOT NULL,
                order_idx         INTEGER NOT NULL,
                prompt            TEXT NOT NULL DEFAULT '',
                element_ids_json  TEXT NOT NULL DEFAULT '[]',
                character_ids_json TEXT NOT NULL DEFAULT '[]',
                location_id       TEXT DEFAULT '',
                camera            TEXT DEFAULT '',
                preview_tos_key   TEXT,
                duration_sec      INTEGER NOT NULL DEFAULT 9,
                video_tos_key     TEXT,
                status            TEXT NOT NULL DEFAULT 'draft',  -- draft | generating | done | failed
                task_id           TEXT,
                seed              INTEGER,
                created_at        REAL NOT NULL,
                updated_at        REAL NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_shots_project ON shots(project_id);
            CREATE INDEX IF NOT EXISTS idx_shots_order   ON shots(project_id, order_idx);

            CREATE TABLE IF NOT EXISTS tracks (
                id           TEXT PRIMARY KEY,
                project_id   TEXT NOT NULL,
                kind         TEXT NOT NULL,      -- tts | bgm | sfx
                tos_key      TEXT NOT NULL,
                start_sec    REAL NOT NULL DEFAULT 0.0,
                volume       REAL NOT NULL DEFAULT 1.0,
                created_at   REAL NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_tracks_project ON tracks(project_id);
            """
        )
        # Additive migrations for pre-existing DBs.
        _maybe_add_col(c, "projects", "settings", "TEXT NOT NULL DEFAULT '{}'")
        _maybe_add_col(c, "projects", "bible", "TEXT NOT NULL DEFAULT '{}'")
        _maybe_add_col(c, "shots", "character_ids_json", "TEXT NOT NULL DEFAULT '[]'")
        _maybe_add_col(c, "shots", "location_id", "TEXT DEFAULT ''")
        _maybe_add_col(c, "shots", "camera", "TEXT DEFAULT ''")
        _maybe_add_col(c, "shots", "preview_tos_key", "TEXT")
        _maybe_add_col(c, "shots", "trim_in", "REAL NOT NULL DEFAULT 0")
        _maybe_add_col(c, "shots", "trim_out", "REAL")
        _maybe_add_col(c, "shots", "transition_in_kind", "TEXT NOT NULL DEFAULT 'cut'")
        _maybe_add_col(c, "shots", "transition_in_dur", "REAL NOT NULL DEFAULT 0.5")


def _maybe_add_col(c, table: str, col: str, spec: str) -> None:
    try:
        c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {spec}")
    except sqlite3.OperationalError:
        pass  # already exists


def _conn():
    assert _DB_PATH, "init_db not called"
    c = sqlite3.connect(_DB_PATH, check_same_thread=False, timeout=10)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    c.row_factory = sqlite3.Row
    return c


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------- projects ----------
DEFAULT_SETTINGS = {
    "ratio": "16:9",         # 16:9 | 9:16 | 1:1 | 4:3 | 3:4 | 21:9
    "resolution": "480p",    # 480p | 720p | 1080p
    "generate_audio": False,
    "watermark": False,
}


DEFAULT_BIBLE = {
    "visual_bible": "",
    "style_image_tos_key": None,  # optional global style reference image
    "characters": [],   # [{"id","name","visual","reference_tos_key"}]
    "locations": [],    # [{"id","name","visual","reference_tos_key"}]
    "assets": [],       # [{"id","name","visual","kind","reference_tos_key"}] — ad-hoc per-shot refs
}


def _hydrate_project(row: dict) -> dict:
    d = dict(row)
    try:
        parsed = json.loads(d.get("settings") or "{}")
    except (ValueError, TypeError):
        parsed = {}
    d["settings"] = {**DEFAULT_SETTINGS, **parsed}
    try:
        b = json.loads(d.get("bible") or "{}")
    except (ValueError, TypeError):
        b = {}
    d["bible"] = {**DEFAULT_BIBLE, **b}
    return d


def create_project(name: str, script: str = "", settings: dict | None = None) -> dict:
    pid = _new_id()
    now = time.time()
    merged = {**DEFAULT_SETTINGS, **(settings or {})}
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT INTO projects(id, name, script, settings, created_at, updated_at) VALUES(?,?,?,?,?,?)",
            (pid, name, script, json.dumps(merged), now, now),
        )
    return get_project(pid)


def list_projects() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
    return [_hydrate_project(r) for r in rows]


def get_project(pid: str) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    return _hydrate_project(r) if r else None


def update_project(pid: str, **fields) -> dict | None:
    allowed = {"name", "script", "settings", "bible"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return get_project(pid)
    # settings: merge partials; bible: replace whole object (simpler semantics).
    if "settings" in fields and isinstance(fields["settings"], dict):
        current = get_project(pid) or {"settings": {}}
        merged = {**current.get("settings", {}), **fields["settings"]}
        fields["settings"] = json.dumps(merged)
    if "bible" in fields and isinstance(fields["bible"], dict):
        fields["bible"] = json.dumps({**DEFAULT_BIBLE, **fields["bible"]})
    sets = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [time.time(), pid]
    with _LOCK, _conn() as c:
        c.execute(f"UPDATE projects SET {sets}, updated_at=? WHERE id=?", vals)
    return get_project(pid)


def delete_project(pid: str) -> None:
    with _LOCK, _conn() as c:
        c.execute("DELETE FROM projects WHERE id=?", (pid,))


# ---------- elements ----------
def create_element(project_id: str, kind: str, name: str, tos_key: str,
                   source: str, prompt: str = "") -> dict:
    eid = _new_id()
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT INTO elements(id, project_id, kind, name, tos_key, source, prompt, created_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (eid, project_id, kind, name, tos_key, source, prompt, time.time()),
        )
    return get_element(eid)


def list_elements(project_id: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM elements WHERE project_id=? ORDER BY created_at ASC",
            (project_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_element(eid: str) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM elements WHERE id=?", (eid,)).fetchone()
    return dict(r) if r else None


def delete_element(eid: str) -> None:
    with _LOCK, _conn() as c:
        c.execute("DELETE FROM elements WHERE id=?", (eid,))


# ---------- shots ----------
def create_shot(project_id: str, order_idx: int, prompt: str = "",
                element_ids: list[str] | None = None,
                character_ids: list[str] | None = None,
                location_id: str = "", camera: str = "",
                duration_sec: int = 9) -> dict:
    sid = _new_id()
    now = time.time()
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT INTO shots(id, project_id, order_idx, prompt, element_ids_json, "
            "character_ids_json, location_id, camera, duration_sec, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (sid, project_id, order_idx, prompt,
             json.dumps(element_ids or []),
             json.dumps(character_ids or []),
             location_id, camera,
             duration_sec, now, now),
        )
    return get_shot(sid)


def _hydrate_shot(row) -> dict:
    d = dict(row)
    d["element_ids"] = json.loads(d.pop("element_ids_json", None) or "[]")
    d["character_ids"] = json.loads(d.pop("character_ids_json", None) or "[]")
    return d


def list_shots(project_id: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM shots WHERE project_id=? ORDER BY order_idx ASC",
            (project_id,),
        ).fetchall()
    return [_hydrate_shot(r) for r in rows]


def get_shot(sid: str) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM shots WHERE id=?", (sid,)).fetchone()
    return _hydrate_shot(r) if r else None


def update_shot(sid: str, **fields) -> dict | None:
    allowed = {"prompt", "duration_sec", "order_idx",
               "element_ids", "character_ids", "location_id", "camera",
               "status", "task_id", "seed", "video_tos_key", "preview_tos_key",
               "trim_in", "trim_out", "transition_in_kind", "transition_in_dur"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return get_shot(sid)
    if "element_ids" in fields:
        fields["element_ids_json"] = json.dumps(fields.pop("element_ids") or [])
    if "character_ids" in fields:
        fields["character_ids_json"] = json.dumps(fields.pop("character_ids") or [])
    sets = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [time.time(), sid]
    with _LOCK, _conn() as c:
        c.execute(f"UPDATE shots SET {sets}, updated_at=? WHERE id=?", vals)
    return get_shot(sid)


def delete_shot(sid: str) -> None:
    with _LOCK, _conn() as c:
        c.execute("DELETE FROM shots WHERE id=?", (sid,))


def migrate_elements_to_bible() -> int:
    """One-time migration: move every project's elements rows into that project's
    bible.assets array, then drop the now-duplicated elements. Idempotent —
    checks for existing asset ids before copying. Returns count of elements moved.
    """
    moved = 0
    with _conn() as c:
        projects = c.execute("SELECT id FROM projects").fetchall()
    for prow in projects:
        pid = prow["id"]
        project = get_project(pid)
        if not project:
            continue
        bible = dict(project.get("bible") or DEFAULT_BIBLE)
        assets = list(bible.get("assets") or [])
        existing_ids = {a.get("id") for a in assets}
        with _conn() as c:
            el_rows = c.execute(
                "SELECT * FROM elements WHERE project_id=? ORDER BY created_at ASC",
                (pid,),
            ).fetchall()
        if not el_rows:
            continue
        for el in el_rows:
            eid = el["id"]
            if eid in existing_ids:
                continue
            assets.append({
                "id": eid,
                "name": el["name"] or "",
                "visual": el["prompt"] or "",
                "kind": el["kind"] or "prop",
                "reference_tos_key": el["tos_key"],
                "from_element": True,
            })
            existing_ids.add(eid)
            moved += 1
        bible["assets"] = assets
        update_project(pid, bible=bible)
        # Now drop the elements rows — their identity lives in bible.assets.
        with _LOCK, _conn() as c:
            c.execute("DELETE FROM elements WHERE project_id=?", (pid,))
    return moved


def replace_shots(project_id: str, shots: list[dict]) -> list[dict]:
    """Bulk replace all shots for a project. Used by auto-storyboard."""
    with _LOCK, _conn() as c:
        c.execute("DELETE FROM shots WHERE project_id=?", (project_id,))
        now = time.time()
        for i, s in enumerate(shots):
            c.execute(
                "INSERT INTO shots(id, project_id, order_idx, prompt, element_ids_json, "
                "character_ids_json, location_id, camera, duration_sec, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (_new_id(), project_id, i,
                 s.get("prompt", ""),
                 json.dumps(s.get("element_ids", [])),
                 json.dumps(s.get("character_ids", [])),
                 s.get("location_id", "") or "",
                 s.get("camera", "") or "",
                 int(s.get("duration_sec", 9)),
                 now, now),
            )
    return list_shots(project_id)
