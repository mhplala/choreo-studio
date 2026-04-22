"""Face detection + masking to satisfy Seedance's privacy filter.

Primary detector: YuNet (OpenCV DNN) — small (~230KB), much better than Haar
at masked faces, profile angles, and low-light. Model is downloaded once on
first use to ~/.cache/choreo-studio/yunet.onnx.

Fallback: Haar cascades (bundled with OpenCV) — used if YuNet model can't be
downloaded (e.g. no network on first run).
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

_YUNET = None
_HAAR_FRONT = None
_HAAR_PROF = None

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_PATH = Path.home() / ".cache" / "choreo-studio" / "yunet.onnx"


def _ensure_yunet():
    global _YUNET
    if _YUNET is not None:
        return _YUNET
    try:
        if not YUNET_PATH.exists():
            YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
        # Input size is overridden per-image via setInputSize.
        _YUNET = cv2.FaceDetectorYN.create(
            str(YUNET_PATH), "", (320, 320),
            score_threshold=0.4, nms_threshold=0.3, top_k=5000,
        )
        return _YUNET
    except Exception:
        _YUNET = False  # sentinel: disable future attempts
        return None


def _haar_detect(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    global _HAAR_FRONT, _HAAR_PROF
    if _HAAR_FRONT is None:
        _HAAR_FRONT = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if _HAAR_PROF is None:
        _HAAR_PROF = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    params = dict(scaleFactor=1.1, minNeighbors=3, minSize=(32, 32))
    faces = list(_HAAR_FRONT.detectMultiScale(gray, **params))
    faces += list(_HAAR_PROF.detectMultiScale(gray, **params))
    faces += [(gray.shape[1] - x - w, y, w, h)
              for (x, y, w, h) in _HAAR_PROF.detectMultiScale(cv2.flip(gray, 1), **params)]
    # Lightweight NMS: drop boxes whose IoU with a kept box > 0.3
    kept: list[tuple[int, int, int, int]] = []
    for (x, y, w, h) in sorted(faces, key=lambda r: -r[2] * r[3]):
        box = (int(x), int(y), int(w), int(h))
        if any(_iou(box, k) > 0.3 for k in kept):
            continue
        kept.append(box)
    return kept


def _iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def detect_faces(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect faces, return list of (x, y, w, h) int tuples in image coords."""
    yunet = _ensure_yunet()
    if yunet:
        h, w = img.shape[:2]
        yunet.setInputSize((w, h))
        _, detections = yunet.detect(img)
        if detections is None:
            return []
        return [(max(0, int(d[0])), max(0, int(d[1])),
                 int(d[2]), int(d[3])) for d in detections]
    # Fallback to Haar
    return _haar_detect(img)


def mask_faces(img_bytes: bytes, pad: float = 0.20) -> tuple[bytes, int]:
    """Detect faces in `img_bytes` and return (masked_png_bytes, face_count).
    If no faces found or decode fails, returns (original_bytes, 0)."""
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return img_bytes, 0
    faces = detect_faces(img)
    if not faces:
        return img_bytes, 0
    h_img, w_img = img.shape[:2]
    for (x, y, w, h) in faces:
        px, py = int(w * pad), int(h * pad)
        x0, y0 = max(0, int(x) - px), max(0, int(y) - py)
        x1, y1 = min(w_img, int(x) + int(w) + px), min(h_img, int(y) + int(h) + py)
        cv2.rectangle(img, (x0, y0), (x1, y1), (90, 95, 110), thickness=-1)
        label = "FACE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, (x1 - x0) / 220.0)
        thickness = max(1, int(scale * 2))
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        tx = x0 + ((x1 - x0) - tw) // 2
        ty = y0 + ((y1 - y0) + th) // 2
        cv2.putText(img, label, (tx, ty), font, scale, (240, 240, 240), thickness, cv2.LINE_AA)
    ok, out = cv2.imencode(".png", img)
    if not ok:
        return img_bytes, len(faces)
    return out.tobytes(), len(faces)
