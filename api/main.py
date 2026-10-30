"""Phase 10: Meter Lens web backend (FastAPI).

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Serves the Inspector at / and the phone app at /app. History is stored in
storage/history.db; set METERLENS_OPEN_BROWSER=0 to skip opening the browser.
"""

import csv
import io
import json
import os
import sqlite3
import sys
import threading
import uuid
import webbrowser
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline import MeterReader  # noqa: E402

WEB_DIR = PROJECT_ROOT / "web" / "app"
INSPECTOR_DIR = PROJECT_ROOT / "web" / "inspector"
STORAGE_DIR = PROJECT_ROOT / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
DB_PATH = STORAGE_DIR / "history.db"
FEEDBACK_LOG = STORAGE_DIR / "feedback_log.csv"
READING_LEN = 5

reader: Optional[MeterReader] = None


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with closing(db()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                image_id TEXT NOT NULL,
                detected INTEGER NOT NULL,
                ai_reading TEXT,
                reading TEXT,
                confidence REAL NOT NULL,
                per_digit_confidence TEXT NOT NULL,
                decision TEXT NOT NULL,
                reasons TEXT,
                sensing_score REAL,
                sensing_flag INTEGER,
                counter_bbox TEXT,
                digit_bboxes TEXT,
                feedback TEXT,
                corrected_reading TEXT,
                feedback_at TEXT
            )""")
        # migrate databases created before the sensing gate
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(readings)")}
        if "reason" in cols and "reasons" not in cols:
            conn.execute("ALTER TABLE readings ADD COLUMN reasons TEXT")
            conn.execute("ALTER TABLE readings ADD COLUMN sensing_score REAL")
            conn.execute("ALTER TABLE readings ADD COLUMN sensing_flag INTEGER")
            for r in conn.execute("SELECT id, reason FROM readings").fetchall():
                reasons = [{"kind": "confidence", "message": r["reason"]}] if r["reason"] else []
                conn.execute("UPDATE readings SET reasons = ? WHERE id = ?", (json.dumps(reasons), r["id"]))
        conn.commit()


def row_to_dict(row) -> dict:
    d = dict(row)
    d["detected"] = bool(d["detected"])
    d["sensing_flag"] = bool(d["sensing_flag"]) if d.get("sensing_flag") is not None else False
    for key in ("per_digit_confidence", "counter_bbox", "digit_bboxes"):
        d[key] = json.loads(d[key]) if d[key] else None
    d["reasons"] = json.loads(d["reasons"]) if d.get("reasons") else []
    d["image_url"] = f"/uploads/{d['image_id']}.jpg"
    return d


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_feedback(history_id: int, image_id: str, predicted: Optional[str], corrected: Optional[str],
                 feedback: str, confidence: float, decision: str):
    new_file = not FEEDBACK_LOG.exists()
    with open(FEEDBACK_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "history_id", "image_id", "predicted_reading", "corrected_reading",
                        "feedback", "confidence_at_prediction", "decision"])
        w.writerow([now_iso(), history_id, image_id, predicted, corrected if feedback == "corrected" else predicted,
                    feedback, f"{confidence:.4f}", decision])


def valid_reading(value: Optional[str]) -> bool:
    return value is not None and len(value) == READING_LEN and value.isdigit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global reader
    init_db()
    reader = MeterReader()
    # load every model at startup
    for m in reader.members:
        reader._model(m)
    _ = reader.detector
    open_browser_when_ready()
    yield


def open_browser_when_ready():
    """Open the app in the default browser shortly after startup."""
    if os.environ.get("METERLENS_OPEN_BROWSER", "1") == "0":
        return
    port = 8000
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == "--port" and i + 1 < len(argv):
            port = int(argv[i + 1])
        elif arg.startswith("--port="):
            port = int(arg.split("=", 1)[1])
    url = f"http://localhost:{port}"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()


app = FastAPI(title="Meter Lens", lifespan=lifespan)


@app.middleware("http")
async def no_cache_pages(request: Request, call_next):
    """Front-end files are edited in place, so ask the browser to revalidate them every load."""
    response = await call_next(request)
    if request.url.path.endswith((".html", ".css", ".js", "/")):
        response.headers["Cache-Control"] = "no-cache"
    return response


class Box(BaseModel):
    x: float
    y: float
    w: float
    h: float


class Reason(BaseModel):
    kind: str            # sensing | confidence | detection
    message: str


class SaveRequest(BaseModel):
    image_id: str
    detected: bool = True
    ai_reading: Optional[str] = None
    confidence: float = 0.0
    per_digit_confidence: list[float] = Field(default_factory=list)
    decision: str
    reasons: list[Reason] = Field(default_factory=list)
    sensing_score: Optional[float] = None
    sensing_flag: bool = False
    counter_bbox: Optional[Box] = None
    digit_bboxes: Optional[list[Box]] = None
    feedback: Optional[str] = None          # confirmed | corrected | None
    corrected_reading: Optional[str] = None


class FeedbackRequest(BaseModel):
    feedback: Optional[str] = None          # confirmed | corrected | None (clear)
    corrected_reading: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "members": [m["name"] for m in reader.members],
            "threshold": reader.threshold, "weights": reader.weights}


async def save_upload(file: UploadFile) -> tuple[str, np.ndarray]:
    """Save an uploaded photo upright and return (image_id, bgr)."""
    raw = await file.read()
    try:
        pil = Image.open(io.BytesIO(raw))
        pil = ImageOps.exif_transpose(pil).convert("RGB")   # phones store rotation in EXIF
    except Exception:
        raise HTTPException(status_code=400, detail="not an image")
    image_id = uuid.uuid4().hex
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    pil.save(UPLOAD_DIR / f"{image_id}.jpg", "JPEG", quality=92)   # stored upright, as the browser will show it
    bgr = np.asarray(pil)[:, :, ::-1].copy()
    return image_id, bgr


def load_upload(image_id: str) -> np.ndarray:
    """Re-load a photo /predict/check already saved, for /predict/recognize."""
    path = UPLOAD_DIR / f"{image_id}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="unknown image_id -- call /predict/check first")
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise HTTPException(status_code=400, detail="stored image could not be read")
    return bgr


class RecognizeRequest(BaseModel):
    image_id: str
    sensing_flag: bool = False   # the value /predict/check already returned for this image_id


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_id, bgr = await save_upload(file)
    result = reader.read(bgr)
    result.update({"image_id": image_id, "image_url": f"/uploads/{image_id}.jpg", "timestamp": now_iso()})
    return result


@app.post("/predict/check")
async def predict_check(file: UploadFile = File(...)):
    image_id, bgr = await save_upload(file)
    result = reader.check(bgr)
    result.update({"image_id": image_id, "image_url": f"/uploads/{image_id}.jpg", "timestamp": now_iso()})
    return result


@app.post("/predict/recognize")
def predict_recognize(req: RecognizeRequest):
    bgr = load_upload(req.image_id)
    result = reader.recognize(bgr, sensing_flag=req.sensing_flag)
    result.update({"image_id": req.image_id, "image_url": f"/uploads/{req.image_id}.jpg", "timestamp": now_iso()})
    return result


@app.get("/history")
def list_history():
    with closing(db()) as conn:
        rows = conn.execute("SELECT * FROM readings ORDER BY id DESC").fetchall()
    return [row_to_dict(r) for r in rows]


@app.get("/history/{item_id}")
def get_history(item_id: int):
    with closing(db()) as conn:
        row = conn.execute("SELECT * FROM readings WHERE id = ?", (item_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="no such reading")
    return row_to_dict(row)


@app.post("/history", status_code=201)
def save_history(req: SaveRequest):
    if not (UPLOAD_DIR / f"{req.image_id}.jpg").exists():
        raise HTTPException(status_code=400, detail="unknown image_id -- call /predict first")
    if req.feedback not in (None, "confirmed", "corrected"):
        raise HTTPException(status_code=400, detail="feedback must be confirmed, corrected or null")
    if req.feedback == "corrected" and not valid_reading(req.corrected_reading):
        raise HTTPException(status_code=400, detail=f"corrected_reading must be {READING_LEN} digits")
    if req.ai_reading is not None and not valid_reading(req.ai_reading):
        raise HTTPException(status_code=400, detail="ai_reading must be 5 digits")
    reading = req.corrected_reading if req.feedback == "corrected" else req.ai_reading
    created = now_iso()
    with closing(db()) as conn:
        cur = conn.execute(
            """INSERT INTO readings (created_at, image_id, detected, ai_reading, reading, confidence,
               per_digit_confidence, decision, reasons, sensing_score, sensing_flag, counter_bbox, digit_bboxes,
               feedback, corrected_reading, feedback_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (created, req.image_id, int(req.detected), req.ai_reading, reading, req.confidence,
             json.dumps(req.per_digit_confidence), req.decision,
             json.dumps([r.model_dump() for r in req.reasons]), req.sensing_score, int(req.sensing_flag),
             json.dumps(req.counter_bbox.model_dump()) if req.counter_bbox else None,
             json.dumps([b.model_dump() for b in req.digit_bboxes]) if req.digit_bboxes else None,
             req.feedback, req.corrected_reading if req.feedback == "corrected" else None,
             created if req.feedback else None))
        conn.commit()
        item_id = cur.lastrowid
        row = conn.execute("SELECT * FROM readings WHERE id = ?", (item_id,)).fetchone()
    if req.feedback:
        log_feedback(item_id, req.image_id, req.ai_reading, req.corrected_reading, req.feedback, req.confidence, req.decision)
    return row_to_dict(row)


@app.patch("/history/{item_id}")
def patch_history(item_id: int, req: FeedbackRequest):
    if req.feedback not in (None, "confirmed", "corrected"):
        raise HTTPException(status_code=400, detail="feedback must be confirmed, corrected or null")
    if req.feedback == "corrected" and not valid_reading(req.corrected_reading):
        raise HTTPException(status_code=400, detail=f"corrected_reading must be {READING_LEN} digits")
    with closing(db()) as conn:
        row = conn.execute("SELECT * FROM readings WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="no such reading")
        corrected = req.corrected_reading if req.feedback == "corrected" else None
        reading = corrected if req.feedback == "corrected" else row["ai_reading"]
        conn.execute("UPDATE readings SET feedback = ?, corrected_reading = ?, reading = ?, feedback_at = ? WHERE id = ?",
                     (req.feedback, corrected, reading, now_iso() if req.feedback else None, item_id))
        conn.commit()
        updated = conn.execute("SELECT * FROM readings WHERE id = ?", (item_id,)).fetchone()
    if req.feedback:
        log_feedback(item_id, row["image_id"], row["ai_reading"], corrected, req.feedback, row["confidence"], row["decision"])
    return row_to_dict(updated)


@app.delete("/history/{item_id}", status_code=204)
def delete_history(item_id: int):
    with closing(db()) as conn:
        cur = conn.execute("DELETE FROM readings WHERE id = ?", (item_id,))
        conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="no such reading")
    return None


@app.get("/app", include_in_schema=False)
def app_root():
    return RedirectResponse("/app/")   # the static mount below only matches with the trailing slash


@app.get("/inspector", include_in_schema=False)
def inspector_root():
    return RedirectResponse("/inspector/")


from api.inspector import router as inspector_router  # noqa: E402  (needs the globals above)
app.include_router(inspector_router)

# Static mounts last, so the API routes above take precedence.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/results", StaticFiles(directory=str(PROJECT_ROOT / "results")), name="results")   # figures and dataset images
if (PROJECT_ROOT / "data").is_dir():                                                           # absent on a hosted demo
    app.mount("/data", StaticFiles(directory=str(PROJECT_ROOT / "data")), name="data")
app.mount("/samples", StaticFiles(directory=str(PROJECT_ROOT / "web" / "samples")), name="samples")   # try-it photos for both UIs
app.mount("/app", StaticFiles(directory=str(WEB_DIR), html=True), name="app")
app.mount("/inspector", StaticFiles(directory=str(INSPECTOR_DIR), html=True), name="inspector-path")   # same portal; the hosted demo can only reach it here
app.mount("/", StaticFiles(directory=str(INSPECTOR_DIR), html=True), name="inspector")
