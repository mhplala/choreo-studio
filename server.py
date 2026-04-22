#!/usr/bin/env python3
"""Choreo Studio — multi-shot AI video creation.

Phase 1 scope:
  - Projects CRUD
  - Elements: upload (file) + generate (Seedream text-to-image)
  - Storyboard: manual shot CRUD + auto-generation via Doubao LLM
  - Video generation (Phase 2) and timeline/audio (Phase 3) come later
"""
import base64
import os
import time
import uuid
from functools import wraps
from pathlib import Path
from typing import Any

import requests
import tos
from flask import Flask, Response, jsonify, request, send_from_directory

import models
import pipeline
import storage

ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
JOBS_DIR = ROOT / "jobs"
DB_PATH = ROOT / "studio.db"
JOBS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------
def load_env():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return {}
    out = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


env = load_env()

AUTH_USER = env.get("AUTH_USER")
AUTH_PASS = env.get("AUTH_PASS")
if not AUTH_USER or not AUTH_PASS:
    raise RuntimeError("AUTH_USER and AUTH_PASS must be set in .env")

ARK_API_KEY = env["ARK_API_KEY"]
SEEDREAM_MODEL = env.get("SEEDREAM_MODEL", "doubao-seedream-3-0-t2i-250415")
LLM_MODEL = env.get("LLM_MODEL", "doubao-1-5-lite-32k-250115")
SEEDANCE_MODEL = env.get("SEEDANCE_MODEL", "doubao-seedance-2-0-fast-260128")

TOS_BUCKET = env["TOS_BUCKET"]
tos_client = tos.TosClientV2(
    env["TOS_ACCESS_KEY"], env["TOS_SECRET_KEY"],
    env["TOS_ENDPOINT"], env["TOS_REGION"],
)

storage.init_db(DB_PATH)
pipeline.configure(tos_client, TOS_BUCKET, JOBS_DIR, ARK_API_KEY, SEEDANCE_MODEL)

# On startup, mark any shots stuck in 'generating' (worker threads died with
# the previous process) as failed so the UI doesn't show a phantom spinner.
for p in storage.list_projects():
    for s in storage.list_shots(p["id"]):
        if s["status"] == "generating":
            storage.update_shot(s["id"], status="failed",
                                task_id="Server restarted while generating")


# ---------------------------------------------------------------------------
# TOS helpers
# ---------------------------------------------------------------------------
def tos_upload_bytes(data: bytes, key: str) -> str:
    tos_client.put_object(TOS_BUCKET, key, content=data)
    return key


def tos_upload_file(local_path: Path, key: str) -> str:
    tos_client.put_object_from_file(TOS_BUCKET, key, str(local_path))
    return key


def tos_presign(key: str, expires: int = 86400) -> str:
    return tos_client.pre_signed_url(
        tos.HttpMethodType.Http_Method_Get, TOS_BUCKET, key, expires=expires
    ).signed_url


# ---------------------------------------------------------------------------
# Flask app + auth
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=str(STATIC_DIR))


def check_auth(hdr: str | None) -> bool:
    if not hdr or not hdr.startswith("Basic "):
        return False
    try:
        user, _, pw = base64.b64decode(hdr.removeprefix("Basic ")).decode().partition(":")
        return user == AUTH_USER and pw == AUTH_PASS
    except Exception:
        return False


def require_auth(fn):
    @wraps(fn)
    def w(*a, **kw):
        if not check_auth(request.headers.get("Authorization")):
            return Response("auth required", 401,
                            {"WWW-Authenticate": 'Basic realm="Choreo Studio"'})
        return fn(*a, **kw)
    return w


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
@require_auth
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/health")
def health():
    return jsonify({"ok": True, "service": "choreo-studio"})


# -- projects --
@app.route("/api/projects", methods=["GET", "POST"])
@require_auth
def projects():
    if request.method == "GET":
        return jsonify(storage.list_projects())
    body = request.get_json() or {}
    name = (body.get("name") or "Untitled project").strip()
    script = body.get("script", "") or ""
    return jsonify(storage.create_project(name=name, script=script))


@app.route("/api/projects/<pid>", methods=["GET", "PATCH", "DELETE"])
@require_auth
def project(pid):
    p = storage.get_project(pid)
    if not p:
        return jsonify({"error": "not found"}), 404
    if request.method == "GET":
        p["elements"] = storage.list_elements(pid)
        p["shots"] = storage.list_shots(pid)
        for e in p["elements"]:
            e["preview_url"] = tos_presign(e["tos_key"], expires=3600)
        for s in p["shots"]:
            if s.get("video_tos_key"):
                s["video_url"] = tos_presign(s["video_tos_key"], expires=3600)
        return jsonify(p)
    if request.method == "DELETE":
        storage.delete_project(pid)
        return jsonify({"ok": True})
    body = request.get_json() or {}
    return jsonify(storage.update_project(pid, **body))


# -- elements: upload --
@app.route("/api/projects/<pid>/elements/upload", methods=["POST"])
@require_auth
def upload_element(pid):
    if not storage.get_project(pid):
        return jsonify({"error": "project not found"}), 404
    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400
    f = request.files["file"]
    kind = request.form.get("kind", "character")
    name = (request.form.get("name") or f.filename or "element").strip()
    # Detect extension from filename, default to jpg
    ext = (Path(f.filename).suffix or ".jpg").lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    eid_hint = uuid.uuid4().hex[:12]
    key = f"dance-gen/studio/projects/{pid}/elements/{eid_hint}{ext}"
    tos_upload_bytes(f.read(), key)
    el = storage.create_element(pid, kind, name, key, "upload", "")
    el["preview_url"] = tos_presign(key, expires=3600)
    return jsonify(el)


# -- elements: generate (Seedream) --
@app.route("/api/projects/<pid>/elements/generate", methods=["POST"])
@require_auth
def generate_element(pid):
    if not storage.get_project(pid):
        return jsonify({"error": "project not found"}), 404
    body = request.get_json() or {}
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    kind = body.get("kind", "character")
    name = (body.get("name") or prompt[:40]).strip()
    size = body.get("size", "1024x1024")
    try:
        url = models.seedream_generate(ARK_API_KEY, SEEDREAM_MODEL, prompt, size=size)
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    img_bytes = requests.get(url, timeout=120).content
    key = f"dance-gen/studio/projects/{pid}/elements/{uuid.uuid4().hex[:12]}.png"
    tos_upload_bytes(img_bytes, key)
    el = storage.create_element(pid, kind, name, key, "generated", prompt)
    el["preview_url"] = tos_presign(key, expires=3600)
    return jsonify(el)


@app.route("/api/projects/<pid>/elements/<eid>", methods=["DELETE"])
@require_auth
def delete_element(pid, eid):
    storage.delete_element(eid)
    return jsonify({"ok": True})


# -- storyboard: auto-generate from script --
@app.route("/api/projects/<pid>/storyboard/auto", methods=["POST"])
@require_auth
def auto_storyboard(pid):
    p = storage.get_project(pid)
    if not p:
        return jsonify({"error": "project not found"}), 404
    body = request.get_json() or {}
    script = (body.get("script") or p.get("script") or "").strip()
    if not script:
        return jsonify({"error": "script is required"}), 400
    hint = body.get("count")
    try:
        shots_data = models.auto_storyboard(ARK_API_KEY, LLM_MODEL, script,
                                            hint_count=int(hint) if hint else None)
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    storage.update_project(pid, script=script)
    shots = storage.replace_shots(pid, shots_data)
    return jsonify({"script": script, "shots": shots})


# -- shots CRUD --
@app.route("/api/projects/<pid>/shots", methods=["POST"])
@require_auth
def create_shot(pid):
    if not storage.get_project(pid):
        return jsonify({"error": "project not found"}), 404
    body = request.get_json() or {}
    existing = storage.list_shots(pid)
    order_idx = body.get("order_idx", len(existing))
    shot = storage.create_shot(
        pid, int(order_idx),
        prompt=body.get("prompt", ""),
        element_ids=body.get("element_ids", []),
        duration_sec=int(body.get("duration_sec", 9)),
    )
    return jsonify(shot)


@app.route("/api/projects/<pid>/shots/<sid>", methods=["GET", "PATCH", "DELETE"])
@require_auth
def shot_detail(pid, sid):
    if request.method == "GET":
        s = storage.get_shot(sid)
        if not s:
            return jsonify({"error": "not found"}), 404
        if s.get("video_tos_key"):
            s["video_url"] = tos_presign(s["video_tos_key"], expires=3600)
        return jsonify(s)
    if request.method == "DELETE":
        storage.delete_shot(sid)
        return jsonify({"ok": True})
    body = request.get_json() or {}
    return jsonify(storage.update_shot(sid, **body))


@app.route("/api/projects/<pid>/shots/<sid>/generate", methods=["POST"])
@require_auth
def generate_shot(pid, sid):
    s = storage.get_shot(sid)
    if not s:
        return jsonify({"error": "shot not found"}), 404
    if s["status"] == "generating":
        return jsonify({"error": "already generating"}), 409
    body = request.get_json() or {}
    chain = bool(body.get("chain_from_prev", False))
    # optimistic status flip so the UI shows spinner immediately
    storage.update_shot(sid, status="generating", task_id=None)
    pipeline.enqueue_shot(pid, sid, chain_from_prev=chain)
    return jsonify({"ok": True, "shot_id": sid, "chain_from_prev": chain}), 202


@app.route("/api/projects/<pid>/shots/<sid>/video")
@require_auth
def shot_video_redirect(pid, sid):
    s = storage.get_shot(sid)
    if not s or not s.get("video_tos_key"):
        return jsonify({"error": "no video yet"}), 404
    return Response("", 302, {"Location": tos_presign(s["video_tos_key"], expires=3600)})


# -- reorder shots --
@app.route("/api/projects/<pid>/shots/reorder", methods=["POST"])
@require_auth
def reorder_shots(pid):
    body = request.get_json() or {}
    order = body.get("order") or []  # list of shot ids in new order
    for idx, sid in enumerate(order):
        storage.update_shot(sid, order_idx=idx)
    return jsonify(storage.list_shots(pid))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=False)
