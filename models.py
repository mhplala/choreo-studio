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


STORYBOARD_SYSTEM = """You are a professional visual director for the Seedance 2.0 video model.
Given a user's idea or script, produce a structured storyboard that follows
Seedance's official 6-step prompt formula:
  Subject -> Action -> Environment -> Camera -> Style -> Constraints
Output is split across a project-level Visual Bible (style, characters,
locations) and per-shot fields.

Output STRICT JSON only (no prose, no markdown fences). Schema:
{
  "visual_bible": "Overall STYLE + lighting tone for the whole project. Include medium (2D cartoon / photoreal / anime / 3D), color palette, film-stock feel, lighting mood (golden hour / overcast / neon / rim-lit). One dense paragraph; this becomes the *style* element of every shot prompt.",
  "characters": [
    {
      "id": "snake_case_id",
      "name": "Readable name",
      "visual": "Physical description: age, ethnicity, hair, clothing, distinctive features. Specific enough for the model to draw consistently across shots; this is the *subject* element of every shot the character appears in."
    }
  ],
  "locations": [
    {
      "id": "snake_case_id",
      "name": "Readable name",
      "visual": "Physical space + dominant lighting + atmosphere. ALWAYS include a lighting cue (e.g. soft natural daylight from large windows, warm tungsten glow, overcast gloom). This is the *environment* element of every shot at this location."
    }
  ],
  "shots": [
    {
      "characters": ["character_id"],
      "location": "location_id or empty string",
      "camera": "ONE of: push-in / pull-out / pan / tracking shot / orbit / aerial / handheld / fixed. Optionally pair with a pacing word (slow, gentle, subtle). NEVER combine multiple movements; NEVER use technical jargon like 24fps f/2.8.",
      "action": "What specifically happens in THIS shot. Use SPECIFIC VERBS with intensity (slowly turns to face the camera, raises right hand) NOT vague adjectives (epic, amazing, beautiful). Don't repeat the character's appearance — that comes from the bible. 1-2 sentences max.",
      "duration_sec": 9
    }
  ]
}

Rules:
- Every id in shots.characters / shots.location MUST be defined in characters[] / locations[].
- Use the same language as the user's input for all free-text fields.
- 3-8 shots for typical inputs; 1 shot if the user clearly describes a single moment.
- Durations: 9 unless user explicitly says otherwise.
- Keep character/location/visual descriptions DETAILED (30-60 words each) — these get injected into every shot's prompt as the Subject and Environment elements, so specificity beats brevity.
- Camera: pick the SINGLE most appropriate movement for that shot's intent. Don't reuse the same one for every shot — vary intentionally.
- Don't reference real people, copyrighted songs/lyrics, or branded IP.
- Don't write 'fast' as a standalone modifier — the official guide warns it destabilizes generation. Use 'quick', 'energetic', or qualify with 'fast but smooth'."""


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()
    return raw


def auto_storyboard(api_key: str, model: str, script: str,
                    hint_count: int | None = None) -> dict:
    """Returns a structured storyboard:
      {"bible": {...}, "shots": [...]}
    where bible has visual_bible/characters/locations and each shot has
    characters/location/camera/action/duration_sec.
    """
    user_msg = script.strip()
    if hint_count:
        user_msg += f"\n\n(Please produce exactly {hint_count} shots.)"
    raw = doubao_chat(
        api_key, model,
        [{"role": "system", "content": STORYBOARD_SYSTEM},
         {"role": "user", "content": user_msg}],
        temperature=0.6, max_tokens=3000,
    )
    raw = _strip_fences(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"doubao returned non-JSON storyboard: {raw[:300]}...") from e

    bible = {
        "visual_bible": (data.get("visual_bible") or "").strip(),
        "characters": [
            {"id": str(c.get("id") or "").strip(),
             "name": (c.get("name") or "").strip(),
             "visual": (c.get("visual") or "").strip()}
            for c in (data.get("characters") or []) if c.get("id")
        ],
        "locations": [
            {"id": str(l.get("id") or "").strip(),
             "name": (l.get("name") or "").strip(),
             "visual": (l.get("visual") or "").strip()}
            for l in (data.get("locations") or []) if l.get("id")
        ],
    }
    known_char_ids = {c["id"] for c in bible["characters"]}
    known_loc_ids = {l["id"] for l in bible["locations"]}

    shots = []
    for s in (data.get("shots") or []):
        action = (s.get("action") or s.get("prompt") or "").strip()
        if not action:
            continue
        shots.append({
            "prompt": action,      # "action" maps to shot.prompt (per-shot specific)
            "character_ids": [c for c in (s.get("characters") or []) if c in known_char_ids],
            "location_id": s.get("location", "") if s.get("location") in known_loc_ids else "",
            "camera": (s.get("camera") or "").strip(),
            "duration_sec": int(s.get("duration_sec", 9)),
        })
    return {"bible": bible, "shots": shots}


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
