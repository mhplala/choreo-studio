# Choreo Studio

Multi-shot AI video creation — storyboard first, generate second.

Inspired by LTX Studio's workflow, powered by Volcengine Ark:
- **Seedream 3.0** for Element (character / style / prop) images
- **Doubao 1.5 Lite** for auto-storyboard (script → N shots)
- **Seedance 2.0 fast** for per-shot video generation (Phase 2)

## Status

**Phase 1** (current):
- [x] Projects CRUD + SQLite persistence
- [x] Elements library (upload + LLM-generate)
- [x] Storyboard: manual shot CRUD + auto-generation from script
- [x] Single-page SPA (vanilla JS, dark theme)
- [x] HTTP Basic Auth

**Phase 2** (next):
- [ ] Per-shot video generation (Seedance)
- [ ] Element reference injection per shot
- [ ] First-frame chaining (optional toggle)
- [ ] Per-shot preview

**Phase 3**:
- [ ] Timeline editor (drag/trim/reorder)
- [ ] TTS per-shot voiceover
- [ ] BGM (aimusicapi integration)
- [ ] Export + transitions
- [ ] Template presets (UI demo / Educational / 短剧 / Meme)

## Quick start

```bash
cp .env.example .env        # fill in TOS_*, ARK_API_KEY, AUTH_USER, AUTH_PASS
pip install -r requirements.txt
python server.py            # http://127.0.0.1:5002
```

## Architecture

```
Browser (SPA)
    │ Basic Auth
    ▼
Flask server.py  (port 5002)
    │
    ├── SQLite (studio.db) — projects / elements / shots / tracks
    ├── TOS — all media under  dance-gen/studio/projects/<project_id>/...
    └── Ark API
          ├── Seedream (image)
          ├── Doubao   (LLM auto-storyboard)
          └── Seedance (video — Phase 2)
```

## Shared infrastructure with Choreo (Dance)

This project is a sibling of [choreo-dance-generator](https://github.com/mhplala/choreo-dance-generator).
They share:
- Same TOS bucket (different subpath)
- Same ECS host (served via nginx path routing — `/` for Studio, `/dance/` for Dance)
- Same Ark API key
