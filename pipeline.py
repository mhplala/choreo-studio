"""Shot generation pipeline.

Given a shot (prompt + element_ids + duration), optionally chained from
the previous shot, submit to Seedance, poll until complete, download the
result, upload to TOS, update DB.

Runs in a background thread. Uses DB for state communication.
"""
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import requests

import models
import storage

JOBS_DIR: Path | None = None
TOS_BUCKET: str | None = None
_TOS_CLIENT = None
_ARK_KEY: str | None = None
_SEEDANCE_MODELS: dict[str, str] = {}


def configure(tos_client, tos_bucket: str, jobs_dir: Path,
              ark_key: str, seedance_models: dict[str, str]):
    """Call once at server startup. `seedance_models` is a dict of tier -> model id,
    with keys 'fast', 'std', 'pro'."""
    global _TOS_CLIENT, TOS_BUCKET, JOBS_DIR, _ARK_KEY, _SEEDANCE_MODELS
    _TOS_CLIENT = tos_client
    TOS_BUCKET = tos_bucket
    JOBS_DIR = jobs_dir
    _ARK_KEY = ark_key
    _SEEDANCE_MODELS = seedance_models


def _pick_model(resolution: str, has_ref: bool) -> str:
    """Return the cheapest Seedance model that can serve the requested
    resolution. Seedance 2.0 has two tiers:
      - fast: 480p only when references are present
      - std:  720p / 1080p and all modes
    """
    if resolution in ("720p", "1080p"):
        return _SEEDANCE_MODELS.get("std") or _SEEDANCE_MODELS["fast"]
    return _SEEDANCE_MODELS["fast"]


# ---------------------------------------------------------------------------
# TOS helpers (thin wrappers so we don't import flask-side globals here)
# ---------------------------------------------------------------------------
def _presign(key: str, expires: int = 86400) -> str:
    import tos
    return _TOS_CLIENT.pre_signed_url(
        tos.HttpMethodType.Http_Method_Get, TOS_BUCKET, key, expires=expires
    ).signed_url


def _upload_bytes(data: bytes, key: str):
    _TOS_CLIENT.put_object(TOS_BUCKET, key, content=data)


def _upload_file(local_path: Path, key: str):
    _TOS_CLIENT.put_object_from_file(TOS_BUCKET, key, str(local_path))


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------
def _extract_last_frame(video_path: Path, out_path: Path):
    # True last frame: seek 1s before EOF, keep decoding, -update 1 overwrites.
    subprocess.run(
        ["ffmpeg", "-y", "-sseof", "-1", "-i", str(video_path),
         "-update", "1", "-q:v", "2", str(out_path)],
        check=True, capture_output=True,
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _build_payload_content(prompt: str, element_urls: list[str],
                           first_frame_url: str | None,
                           style_video_url: str | None = None) -> list[dict]:
    content = [{"type": "text", "text": prompt}]
    if first_frame_url:
        content.append({"type": "image_url",
                        "image_url": {"url": first_frame_url},
                        "role": "reference_image"})
    for url in element_urls:
        content.append({"type": "image_url",
                        "image_url": {"url": url},
                        "role": "reference_image"})
    if style_video_url:
        content.append({"type": "video_url",
                        "video_url": {"url": style_video_url},
                        "role": "reference_video"})
    return content


def bible_reference_keys(bible: dict, shot: dict) -> list[str]:
    """Collect TOS keys for any reference images attached to the characters
    and location that appear in this shot. Used so Seedance gets the user-
    curated visual anchors automatically — no per-shot manual attaching."""
    keys: list[str] = []
    char_ids = set(shot.get("character_ids") or [])
    for c in (bible.get("characters") or []):
        if c.get("id") in char_ids and c.get("reference_tos_key"):
            keys.append(c["reference_tos_key"])
    loc_id = shot.get("location_id") or ""
    if loc_id:
        for l in (bible.get("locations") or []):
            if l.get("id") == loc_id and l.get("reference_tos_key"):
                keys.append(l["reference_tos_key"])
                break
    return keys


def compose_prompt(bible: dict, shot: dict) -> str:
    """Build the full Seedance prompt from the project's bible and this shot's
    structured fields. This is the core of cross-shot continuity: every shot's
    prompt carries the same visual_bible + character descriptions + location
    description, so Seedance sees coherent context across the whole project.
    """
    parts = []
    vb = (bible.get("visual_bible") or "").strip()
    if vb:
        parts.append(vb)

    char_ids = shot.get("character_ids") or []
    chars = [c for c in (bible.get("characters") or []) if c["id"] in char_ids]
    if chars:
        parts.append(
            "角色: " + "；".join(
                f"{c['name']}（{c['visual']}）" if c.get("visual") else c.get("name", "")
                for c in chars
            )
        )

    loc_id = shot.get("location_id") or ""
    loc = next((l for l in (bible.get("locations") or []) if l["id"] == loc_id), None)
    if loc and loc.get("visual"):
        parts.append(f"场景: {loc['visual']}")

    cam = (shot.get("camera") or "").strip()
    if cam:
        parts.append(f"镜头: {cam}")

    action = (shot.get("prompt") or "").strip()
    if action:
        parts.append(f"动作: {action}")

    return "\n\n".join(parts) if parts else "A cinematic scene."


def _run_one_shot(project_id: str, shot_id: str, chain_from_prev: bool):
    """Worker body. Updates shot.status in DB as it progresses."""
    shot = storage.get_shot(shot_id)
    if not shot:
        return
    project = storage.get_project(project_id)
    work = JOBS_DIR / f"{project_id}_{shot_id}"
    work.mkdir(exist_ok=True, parents=True)
    tos_prefix = f"dance-gen/studio/projects/{project_id}/shots/{shot_id}"

    try:
        storage.update_shot(shot_id, status="generating", task_id=None)

        # Resolve element reference images
        element_ids = shot.get("element_ids", [])
        element_urls = []
        for eid in element_ids:
            el = storage.get_element(eid)
            if el:
                element_urls.append(_presign(el["tos_key"]))

        # Auto-attach bible reference images for the characters/location this
        # shot uses. This is the core payoff of the Visual Bible: one click
        # generates refs once, every shot that mentions that character or
        # location automatically gets them attached.
        bible = (project or {}).get("bible") or {}
        bible_ref_urls = [_presign(k) for k in bible_reference_keys(bible, shot)]

        # The Seedream-generated preview (if any) locks composition: attach as
        # first-frame reference. Strongly outranks chain mode when both exist.
        first_frame_url = None
        if shot.get("preview_tos_key"):
            first_frame_url = _presign(shot["preview_tos_key"])

        # Optionally chain: extract last frame of previous shot's video. Only
        # if we didn't already have a preview first-frame.
        if not first_frame_url and chain_from_prev:
            shots = storage.list_shots(project_id)
            prev = None
            for s in shots:
                if s["id"] == shot_id:
                    break
                prev = s
            if prev and prev.get("video_tos_key") and prev["status"] == "done":
                # Download prev video, extract last frame, upload as reference
                prev_video_url = _presign(prev["video_tos_key"])
                prev_local = work / "prev.mp4"
                with requests.get(prev_video_url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(prev_local, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                ff_local = work / "first_frame.jpg"
                _extract_last_frame(prev_local, ff_local)
                ff_key = f"{tos_prefix}/first_frame.jpg"
                _upload_file(ff_local, ff_key)
                first_frame_url = _presign(ff_key)

        # Seedance's fast model has different duration constraints depending on
        # whether we give it any references:
        #   - pure t2v (no refs): only 5 or 10 seconds accepted
        #   - i2v / r2v (with image or video ref): 5-12 seconds
        has_ref = bool(element_urls) or bool(first_frame_url) or bool(bible_ref_urls)
        requested = int(shot.get("duration_sec", 9))
        if has_ref:
            duration = max(5, min(12, requested))
        else:
            duration = 5 if requested <= 7 else 10

        # Pull project-level generation params from settings.
        settings = (project or {}).get("settings", {}) or {}
        ratio = settings.get("ratio", "16:9")
        resolution = settings.get("resolution", "480p")
        gen_audio = bool(settings.get("generate_audio", False))

        # Auto-pick the cheapest Seedance model tier that supports the
        # requested resolution — user doesn't need to know fast/std exists.
        model_id = _pick_model(resolution, has_ref)

        # Compose the full prompt from project bible + shot structured fields,
        # so cross-shot continuity (character, location, style) is baked in.
        full_prompt = compose_prompt(bible, shot)

        # Reference image order matters: first_frame (if any) locks composition
        # most strongly, then bible refs (style/character/location anchors),
        # then user-attached elements. Cap at 4 to stay well under Seedance's
        # internal limit on reference counts.
        all_refs = ([first_frame_url] if first_frame_url else []) + bible_ref_urls + element_urls
        all_refs = all_refs[:4]

        task_id = models.seedance_submit(
            _ARK_KEY, model_id,
            prompt=full_prompt,
            image_urls=all_refs,
            ratio=ratio, resolution=resolution,
            duration=duration, generate_audio=gen_audio,
        )
        storage.update_shot(shot_id, task_id=task_id)

        # Poll
        def _on(status):
            cur = storage.get_shot(shot_id)
            if cur:
                storage.update_shot(shot_id, task_id=task_id)
        result = models.seedance_poll(_ARK_KEY, task_id, on_status=_on)
        if result.get("status") != "succeeded":
            raise RuntimeError(f"Seedance task failed: {result.get('status')} {result}")

        # Download result, upload to TOS
        video_url = result["content"]["video_url"]
        local_video = work / "out.mp4"
        with requests.get(video_url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(local_video, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        video_key = f"{tos_prefix}/video_{uuid.uuid4().hex[:8]}.mp4"
        _upload_file(local_video, video_key)

        storage.update_shot(shot_id, status="done",
                            video_tos_key=video_key,
                            seed=result.get("seed"))
    except Exception as e:
        storage.update_shot(shot_id, status="failed", task_id=str(e)[:200])


# Public: kick off video generation in a background thread. Returns immediately.
def enqueue_shot(project_id: str, shot_id: str, chain_from_prev: bool = False):
    t = threading.Thread(
        target=_run_one_shot,
        args=(project_id, shot_id, chain_from_prev),
        daemon=True,
    )
    t.start()


# ---------------------------------------------------------------------------
# Storyboard still previews (Seedream) — much cheaper than video, catches
# composition/continuity problems before burning video credits.
# ---------------------------------------------------------------------------
def _seedream_size_for_ratio(ratio: str) -> str:
    return {
        "1:1":  "2048x2048",
        "9:16": "1536x2730",
        "16:9": "2730x1536",
        "3:4":  "1792x2304",
        "4:3":  "2304x1792",
        "21:9": "2730x1536",  # closest available that clears the 3.7M floor
    }.get(ratio, "2048x2048")


def generate_shot_preview(project_id: str, shot_id: str,
                          seedream_model: str) -> str | None:
    """Generate a Seedream still for this shot using the composed prompt,
    upload it to TOS, attach as shot.preview_tos_key. Returns the TOS key
    or None on failure.
    """
    shot = storage.get_shot(shot_id)
    project = storage.get_project(project_id)
    if not shot or not project:
        return None

    bible = project.get("bible") or {}
    full_prompt = compose_prompt(bible, shot)
    ratio = (project.get("settings") or {}).get("ratio", "16:9")
    size = _seedream_size_for_ratio(ratio)

    try:
        url = models.seedream_generate(_ARK_KEY, seedream_model,
                                       prompt=full_prompt, size=size)
    except Exception:
        return None

    try:
        img_bytes = requests.get(url, timeout=120).content
    except Exception:
        return None

    key = f"dance-gen/studio/projects/{project_id}/shots/{shot_id}/preview.png"
    _TOS_CLIENT.put_object(TOS_BUCKET, key, content=img_bytes)
    storage.update_shot(shot_id, preview_tos_key=key)
    return key


def enqueue_preview(project_id: str, shot_id: str, seedream_model: str):
    """Run preview gen in a background thread (Seedream is ~5-15s so this
    keeps the request hot-path snappy)."""
    def _runner():
        generate_shot_preview(project_id, shot_id, seedream_model)
    threading.Thread(target=_runner, daemon=True).start()


# ---------------------------------------------------------------------------
# Export: stitch all done shots in order into a single MP4 via ffmpeg concat
# ---------------------------------------------------------------------------
def export_project(project_id: str) -> dict:
    """Concat all done shots (in order_idx order) into one final.mp4, upload
    to TOS, record on the project. Returns {'tos_key': ..., 'duration': ...,
    'shot_count': N} on success, or raises.
    """
    shots = [s for s in storage.list_shots(project_id)
             if s.get("status") == "done" and s.get("video_tos_key")]
    if not shots:
        raise RuntimeError("No done shots to export")

    work = JOBS_DIR / f"export_{project_id}_{int(time.time())}"
    work.mkdir(parents=True, exist_ok=True)

    # Download each shot's video to a local file, record ordered list.
    local_paths: list[Path] = []
    for i, s in enumerate(shots):
        url = _presign(s["video_tos_key"])
        local = work / f"seg_{i:03d}.mp4"
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(local, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        local_paths.append(local)

    # Write ffmpeg concat list (relative filenames so ffmpeg finds them in -cwd).
    concat_txt = work / "concat.txt"
    concat_txt.write_text("".join(f"file '{p.name}'\n" for p in local_paths))

    final_local = work / "final.mp4"
    # Use concat demuxer with re-encode to survive mismatched codecs/resolutions.
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-c:a", "aac", "-b:a", "192k",
         "-movflags", "+faststart",
         str(final_local)],
        check=True, capture_output=True, cwd=str(work),
    )

    # Probe duration, upload to TOS.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(final_local)],
        check=True, capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip() or 0)

    key = f"dance-gen/studio/projects/{project_id}/exports/final_{int(time.time())}.mp4"
    _TOS_CLIENT.put_object_from_file(TOS_BUCKET, key, str(final_local))

    # Clean up local working dir (optional — leave for debug / allow disk cleanup).
    # shutil.rmtree(work, ignore_errors=True)

    return {"tos_key": key, "duration": duration, "shot_count": len(shots)}


_EXPORT_STATE: dict[str, dict] = {}  # project_id -> {'status', 'result', 'error'}


def enqueue_export(project_id: str):
    """Kick off export in a background thread; frontend polls
    GET /api/projects/:pid/export/status to observe progress."""
    _EXPORT_STATE[project_id] = {"status": "running", "started_at": time.time()}

    def _runner():
        try:
            result = export_project(project_id)
            _EXPORT_STATE[project_id] = {
                "status": "done",
                "result": result,
                "finished_at": time.time(),
            }
        except Exception as e:
            _EXPORT_STATE[project_id] = {
                "status": "failed",
                "error": str(e),
                "finished_at": time.time(),
            }
    threading.Thread(target=_runner, daemon=True).start()


def get_export_state(project_id: str) -> dict:
    return _EXPORT_STATE.get(project_id, {"status": "idle"})
