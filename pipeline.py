"""Shot generation pipeline.

Given a shot (prompt + element_ids + duration), optionally chained from
the previous shot, submit to Seedance, poll until complete, download the
result, upload to TOS, update DB.

Runs in a background thread. Uses DB for state communication.
"""
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
_SEEDANCE_MODEL: str | None = None


def configure(tos_client, tos_bucket: str, jobs_dir: Path,
              ark_key: str, seedance_model: str):
    """Call once at server startup."""
    global _TOS_CLIENT, TOS_BUCKET, JOBS_DIR, _ARK_KEY, _SEEDANCE_MODEL
    _TOS_CLIENT = tos_client
    TOS_BUCKET = tos_bucket
    JOBS_DIR = jobs_dir
    _ARK_KEY = ark_key
    _SEEDANCE_MODEL = seedance_model


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

        # Optionally chain: extract last frame of previous shot's video
        first_frame_url = None
        if chain_from_prev:
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
        has_ref = bool(element_urls) or bool(first_frame_url)
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

        task_id = models.seedance_submit(
            _ARK_KEY, _SEEDANCE_MODEL,
            prompt=shot["prompt"] or "A cinematic scene.",
            image_urls=([first_frame_url] if first_frame_url else []) + element_urls,
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


# Public: kick off generation in a background thread. Returns immediately.
def enqueue_shot(project_id: str, shot_id: str, chain_from_prev: bool = False):
    t = threading.Thread(
        target=_run_one_shot,
        args=(project_id, shot_id, chain_from_prev),
        daemon=True,
    )
    t.start()
