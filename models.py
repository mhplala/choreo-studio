"""Wrappers around Volcengine Ark APIs we depend on:
- Seedream 3.0 (text-to-image) for Elements
- Doubao 1.5 Lite (chat) for auto-storyboard
- Seedance 2.0 fast (image+video-to-video) for shot rendering

All three sit behind the same ARK_API_KEY and base URL:
  https://ark.cn-beijing.volces.com/api/v3/*
"""
import json
import time
from typing import Any

import requests

ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"


def _retry(fn, *args, **kwargs):
    for attempt in range(5):
        try:
            return fn(*args, **kwargs)
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError,
                requests.exceptions.Timeout):
            time.sleep(4 * (attempt + 1))
    raise RuntimeError("ark api retries exhausted")


# ---------------------------------------------------------------------------
# Seedream — text to image (for Elements auto-generation)
# ---------------------------------------------------------------------------
def seedream_generate(api_key: str, model: str, prompt: str,
                      size: str = "1024x1024", seed: int | None = None) -> str:
    """Returns an image URL (signed, short-lived). Caller must download & re-upload."""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    body: dict[str, Any] = {"model": model, "prompt": prompt, "size": size,
                            "response_format": "url", "watermark": False}
    if seed is not None:
        body["seed"] = seed
    r = _retry(lambda: requests.post(f"{ARK_BASE}/images/generations",
                                     headers=headers, json=body, timeout=90))
    if r.status_code >= 400:
        raise RuntimeError(f"seedream {r.status_code}: {r.text}")
    data = r.json()
    return data["data"][0]["url"]


# ---------------------------------------------------------------------------
# Doubao LLM — chat completion (for auto-storyboard)
# ---------------------------------------------------------------------------
def doubao_chat(api_key: str, model: str, messages: list[dict],
                temperature: float = 0.7, max_tokens: int = 2000) -> str:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    body = {"model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    r = _retry(lambda: requests.post(f"{ARK_BASE}/chat/completions",
                                     headers=headers, json=body, timeout=60))
    if r.status_code >= 400:
        raise RuntimeError(f"doubao {r.status_code}: {r.text}")
    return r.json()["choices"][0]["message"]["content"]


STORYBOARD_SYSTEM = (
    "You are a professional video director. Given a user's idea or script, break it down "
    "into a storyboard of individual shots for an AI video generator. "
    "Output STRICT JSON only (no prose, no markdown fences) in this shape:\n"
    '{"shots": [{"prompt": "...", "duration_sec": 9}, ...]}\n'
    "Rules:\n"
    "- Each shot's prompt should be a vivid visual description (subject, action, camera, mood, style) "
    "in the same language as the user's input.\n"
    "- Each shot is 9 seconds by default; only use a different duration if the user explicitly requests it.\n"
    "- 3-8 shots for typical inputs; 1 shot if the input clearly describes a single moment.\n"
    "- Ensure cross-shot character/style continuity in the descriptions.\n"
    "- No lyrics, no copyrighted song references, no real-person likeness requests."
)


def auto_storyboard(api_key: str, model: str, script: str,
                    hint_count: int | None = None) -> list[dict]:
    user_msg = script.strip()
    if hint_count:
        user_msg += f"\n\n(Please produce exactly {hint_count} shots.)"
    raw = doubao_chat(
        api_key, model,
        [{"role": "system", "content": STORYBOARD_SYSTEM},
         {"role": "user", "content": user_msg}],
        temperature=0.7, max_tokens=2000,
    )
    # Be forgiving: strip potential markdown fences.
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"doubao returned non-JSON storyboard: {raw[:300]}...") from e
    shots = data.get("shots", [])
    # Normalize
    return [{"prompt": s.get("prompt", "").strip(),
             "duration_sec": int(s.get("duration_sec", 9))} for s in shots if s.get("prompt")]


# ---------------------------------------------------------------------------
# Seedance — image-to-video (for per-shot rendering) — Phase 2
# ---------------------------------------------------------------------------
def seedance_submit(api_key: str, model: str, prompt: str,
                    image_urls: list[str] | None = None,
                    video_url: str | None = None,
                    ratio: str = "1:1", resolution: str = "480p",
                    duration: int = 9, generate_audio: bool = False) -> str:
    content: list[dict] = [{"type": "text", "text": prompt}]
    for url in (image_urls or []):
        content.append({"type": "image_url", "image_url": {"url": url},
                        "role": "reference_image"})
    if video_url:
        content.append({"type": "video_url", "video_url": {"url": video_url},
                        "role": "reference_video"})
    body = {"model": model, "content": content, "ratio": ratio,
            "resolution": resolution, "duration": duration,
            "generate_audio": generate_audio, "watermark": False}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    r = _retry(lambda: requests.post(f"{ARK_BASE}/contents/generations/tasks",
                                     headers=headers, json=body, timeout=60))
    if r.status_code >= 400:
        raise RuntimeError(f"seedance submit {r.status_code}: {r.text}")
    return r.json()["id"]


def seedance_poll(api_key: str, task_id: str, on_status=None) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{ARK_BASE}/contents/generations/tasks/{task_id}"
    while True:
        r = _retry(lambda: requests.get(url, headers=headers, timeout=30))
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        if on_status:
            on_status(status)
        if status in ("succeeded", "failed", "cancelled"):
            return data
        time.sleep(8)
