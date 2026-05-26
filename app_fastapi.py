"""
Interview.AI face verification backend.

FastAPI is the only supported server implementation.
"""

import json
import os
import time

import cv2
import numpy as np
from deepface import DeepFace
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from admin_db import admin_db


app = FastAPI(title="Interview.AI Face Verification Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("logs", exist_ok=True)

MODEL = "ArcFace"
DETECTOR = "mtcnn"
ARCFACE_THRESHOLD = 0.65

MIN_BRIGHTNESS = 20
MAX_BRIGHTNESS = 242
MIN_FACE_PX = 60
FACE_CENTER_RATIO = 0.12
EYE_GAP_RATIO = 0.12

MAX_WARNINGS = 3
MAX_FAIL_STREAK = 3

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

candidate_store = {}
session_state = {}


def api_error(message, status_code=400):
    return JSONResponse({"error": message}, status_code=status_code)


async def decode_uploaded_image(file: UploadFile):
    file_bytes = np.frombuffer(await file.read(), np.uint8)
    if file_bytes.size == 0:
        return None
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def cosine_distance(a, b):
    a, b = np.array(a), np.array(b)
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def calculate_confidence(distance):
    confidence = (1 - distance / ARCFACE_THRESHOLD) * 100
    return round(max(0.0, min(100.0, confidence)), 2)


def get_status(confidence):
    if confidence >= 80:
        return "Strong Match"
    if confidence >= 55:
        return "Possible Match"
    if confidence >= 40:
        return "Weak Match"
    return "No Match"


def log_check(candidate_id, result):
    log_path = f"logs/{candidate_id}.json"
    logs = []
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(result)
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)


def check_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))

    if brightness < MIN_BRIGHTNESS:
        return False, brightness, "too_dark"
    if brightness > MAX_BRIGHTNESS:
        return False, brightness, "too_bright"
    return True, brightness, "ok"


def check_face_pose(img):
    try:
        results = DeepFace.represent(
            img_path=img,
            model_name=MODEL,
            detector_backend=DETECTOR,
            enforce_detection=False,
        )
        if not results:
            return False, []

        valid = [r for r in results if r.get("face_confidence", 1.0) >= 0.70]
        if not valid:
            return False, []
        return True, valid
    except Exception as e:
        print(f"[check_face_pose] Exception: {e}")
        return False, []


def build_result(
    candidate_id,
    state,
    status,
    confidence,
    distance,
    verified,
    terminate,
    reason,
    brightness=None,
):
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_id": candidate_id,
        "status": status,
        "confidence": confidence,
        "distance": round(distance, 4) if distance is not None else None,
        "verified": verified,
        "warning_count": state["warning_count"],
        "fail_streak": state["fail_streak"],
        "check_count": state["check_count"],
        "terminate": terminate,
        "reason": reason,
        "brightness": round(brightness, 1) if brightness is not None else None,
    }


def ensure_session_state(candidate_id):
    if candidate_id not in session_state:
        session_state[candidate_id] = {
            "warning_count": 0,
            "fail_streak": 0,
            "terminated": False,
            "check_count": 0,
        }
    return session_state[candidate_id]


def apply_warning_termination(candidate_id, result, img=None):
    state = session_state[candidate_id]
    if state["warning_count"] >= MAX_WARNINGS:
        state["terminated"] = True
        result["terminate"] = True
        result["reason"] = "Too many warnings"
        log_check(candidate_id, result)
        admin_db.delete_reference_photo(candidate_id)
        if img is not None:
            term_path = os.path.join(UPLOAD_FOLDER, f"termination_{candidate_id}.jpg")
            cv2.imwrite(term_path, img)
            admin_db.store_evidence(candidate_id, term_path, "termination_photo")
            if os.path.exists(term_path):
                os.remove(term_path)


@app.get("/health")
def health():
    return {
        "status": "running",
        "model": MODEL,
        "detector": DETECTOR,
        "threshold": ARCFACE_THRESHOLD,
        "registered_candidates": len(candidate_store),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.post("/start_interview")
async def start_interview(
    candidate_id: str = Form(None),
    candidate_name: str = Form(None),
    file: UploadFile = File(None),
):
    print(f"[start_interview] candidate_id={candidate_id!r}, candidate_name={candidate_name!r}, file={file}")
    if file:
        print(f"[start_interview] file.filename={file.filename!r}, file.content_type={file.content_type!r}")

    if not candidate_id or not candidate_name:
        print(f"[start_interview] REJECTED: Missing candidate_id or candidate_name")
        return api_error("Missing candidate_id or candidate_name")

    if file is None:
        print(f"[start_interview] REJECTED: No reference image provided")
        return api_error("No reference image provided")

    if file.filename == "":
        print(f"[start_interview] REJECTED: Empty filename")
        return api_error("Empty filename")

    profile_filename = f"reference_{candidate_id}.jpg"
    profile_path = os.path.join(UPLOAD_FOLDER, profile_filename)
    with open(profile_path, "wb") as profile_file:
        profile_file.write(await file.read())

    try:
        results = DeepFace.represent(
            img_path=profile_path,
            model_name=MODEL,
            detector_backend=DETECTOR,
            enforce_detection=False,
        )
    except ValueError:
        os.remove(profile_path)
        return api_error("No face found in reference image")
    except Exception as e:
        os.remove(profile_path)
        return api_error(f"Failed to process reference image: {str(e)}")

    if len(results) == 0:
        os.remove(profile_path)
        return api_error("No face detected in reference image")

    if len(results) > 1:
        os.remove(profile_path)
        return api_error("Multiple faces in reference image")

    facial_area = results[0].get("facial_area", {})
    face_w = facial_area.get("w", 0)
    face_h = facial_area.get("h", 0)

    if face_w < 80 or face_h < 80:
        os.remove(profile_path)
        return api_error("Face in reference image is too small")

    candidate_store[candidate_id] = {
        "embedding": results[0]["embedding"],
        "profile_image": profile_filename,
        "candidate_name": candidate_name,
        "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "face_size": f"{face_w}x{face_h}px",
    }

    session_state[candidate_id] = {
        "warning_count": 0,
        "fail_streak": 0,
        "terminated": False,
        "check_count": 0,
    }

    # Persist to admin database
    admin_db.save_candidate(candidate_id, {
        "candidate_name": candidate_name,
        "face_size": f"{face_w}x{face_h}px",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "Active",
    })

    return {
        "success": True,
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "message": "Interview started. Monitoring enabled.",
        "face_size": f"{face_w}x{face_h}px",
    }


@app.post("/verify_frame")
async def verify_frame(candidate_id: str = Form(None), file: UploadFile = File(None)):
    if not candidate_id:
        return api_error("Missing candidate_id")

    if candidate_id not in candidate_store:
        return api_error("Candidate not registered")

    if file is None:
        return api_error("No frame provided")

    state = ensure_session_state(candidate_id)

    if state["terminated"]:
        return {
            "terminate": True,
            "reason": "Session already terminated",
            "status": "terminated",
            "verified": False,
            "confidence": 0,
            "warning_count": state["warning_count"],
            "fail_streak": state["fail_streak"],
            "check_count": state["check_count"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    timings = {}
    request_start = time.perf_counter()
    state["check_count"] += 1

    decode_start = time.perf_counter()
    img = await decode_uploaded_image(file)
    timings["decode_ms"] = round((time.perf_counter() - decode_start) * 1000, 1)
    if img is None:
        return api_error("Failed to read frame")

    frame_h, frame_w = img.shape[:2]
    terminate = False
    reason = None

    brightness_start = time.perf_counter()
    bright_ok, brightness_val, bright_label = check_brightness(img)
    timings["brightness_ms"] = round((time.perf_counter() - brightness_start) * 1000, 1)

    if not bright_ok:
        state["warning_count"] += 1
        status = "too_dark" if bright_label == "too_dark" else "too_bright"
        result = build_result(
            candidate_id,
            state,
            status=status,
            confidence=0,
            distance=None,
            verified=False,
            terminate=terminate,
            reason=reason,
            brightness=brightness_val,
        )
        timings["total_ms"] = round((time.perf_counter() - request_start) * 1000, 1)
        result["timings"] = timings
        log_check(candidate_id, result)
        apply_warning_termination(candidate_id, result, img)
        result["suggestions"] = [
            "Turn on more lights" if bright_label == "too_dark" else "Reduce glare/brightness"
        ]
        print(json.dumps(result, indent=2))
        return result

    face_start = time.perf_counter()
    face_found, results = check_face_pose(img)
    timings["deepface_ms"] = round((time.perf_counter() - face_start) * 1000, 1)

    if not face_found or len(results) == 0:
        state["warning_count"] += 1
        result = build_result(
            candidate_id,
            state,
            status="no_face",
            confidence=0,
            distance=None,
            verified=False,
            terminate=terminate,
            reason=reason,
            brightness=brightness_val,
        )
        timings["total_ms"] = round((time.perf_counter() - request_start) * 1000, 1)
        result["timings"] = timings
        log_check(candidate_id, result)
        apply_warning_termination(candidate_id, result, img)
        result["suggestions"] = ["Ensure your face is visible to the camera"]
        print(json.dumps(result, indent=2))
        return result

    if len(results) > 1:
        state["warning_count"] += 1
        result = build_result(
            candidate_id,
            state,
            status="multiple_faces",
            confidence=0,
            distance=None,
            verified=False,
            terminate=terminate,
            reason=reason,
            brightness=brightness_val,
        )
        timings["total_ms"] = round((time.perf_counter() - request_start) * 1000, 1)
        result["timings"] = timings
        log_check(candidate_id, result)
        apply_warning_termination(candidate_id, result, img)
        result["suggestions"] = ["Only the candidate should be visible to the camera"]
        print(json.dumps(result, indent=2))
        return result

    facial_area = results[0].get("facial_area", {})
    face_w = facial_area.get("w", 0)
    face_h = facial_area.get("h", 0)
    face_x = facial_area.get("x", 0)
    face_y = facial_area.get("y", 0)

    center_x = face_x + face_w / 2
    center_y = face_y + face_h / 2

    too_small = face_w < MIN_FACE_PX or face_h < MIN_FACE_PX
    too_left = center_x < frame_w * FACE_CENTER_RATIO
    too_right = center_x > frame_w * (1 - FACE_CENTER_RATIO)
    too_up = center_y < frame_h * FACE_CENTER_RATIO
    too_down = center_y > frame_h * (1 - FACE_CENTER_RATIO)

    if too_small or too_left or too_right or too_up or too_down:
        state["warning_count"] += 1
        result = build_result(
            candidate_id,
            state,
            status="face_not_centered",
            confidence=0,
            distance=None,
            verified=False,
            terminate=terminate,
            reason=reason,
            brightness=brightness_val,
        )
        timings["total_ms"] = round((time.perf_counter() - request_start) * 1000, 1)
        result["timings"] = timings
        log_check(candidate_id, result)
        apply_warning_termination(candidate_id, result, img)

        suggestions = []
        if too_small:
            suggestions.append("Move closer to camera")
        if too_left or too_right:
            suggestions.append("Center your face horizontally")
        if too_up or too_down:
            suggestions.append("Center your face vertically")
        result["suggestions"] = suggestions
        print(json.dumps(result, indent=2))
        return result

    landmarks = results[0].get("facial_area", {})
    left_eye = landmarks.get("left_eye")
    right_eye = landmarks.get("right_eye")

    side_face = False
    if left_eye and right_eye and face_w > 0:
        eye_gap = abs(right_eye[0] - left_eye[0])
        side_face = eye_gap < EYE_GAP_RATIO * face_w

    if side_face:
        state["warning_count"] += 1
        result = build_result(
            candidate_id,
            state,
            status="side_face",
            confidence=0,
            distance=None,
            verified=False,
            terminate=terminate,
            reason=reason,
            brightness=brightness_val,
        )
        timings["total_ms"] = round((time.perf_counter() - request_start) * 1000, 1)
        result["timings"] = timings
        log_check(candidate_id, result)
        apply_warning_termination(candidate_id, result, img)
        result["suggestions"] = ["Look directly at the camera"]
        print(json.dumps(result, indent=2))
        return result

    webcam_embedding = results[0]["embedding"]
    stored_embedding = candidate_store[candidate_id]["embedding"]

    distance = cosine_distance(stored_embedding, webcam_embedding)
    confidence = calculate_confidence(distance)
    status = get_status(confidence)
    verified = distance < ARCFACE_THRESHOLD

    if verified and confidence >= 75:
        state["fail_streak"] = 0
    elif verified and confidence >= 45:
        state["fail_streak"] = max(0, state["fail_streak"] - 1)
    else:
        state["fail_streak"] += 1

    if state["fail_streak"] >= MAX_FAIL_STREAK:
        state["terminated"] = True
        terminate = True
        reason = "Identity verification failed repeatedly"
        admin_db.delete_reference_photo(candidate_id)
        term_path = os.path.join(UPLOAD_FOLDER, f"termination_{candidate_id}.jpg")
        cv2.imwrite(term_path, img)
        admin_db.store_evidence(candidate_id, term_path, "termination_photo")
        if os.path.exists(term_path):
            os.remove(term_path)

    if state["warning_count"] >= MAX_WARNINGS:
        state["terminated"] = True
        terminate = True
        reason = "Too many warnings"
        admin_db.delete_reference_photo(candidate_id)
        term_path = os.path.join(UPLOAD_FOLDER, f"termination_{candidate_id}.jpg")
        cv2.imwrite(term_path, img)
        admin_db.store_evidence(candidate_id, term_path, "termination_photo")
        if os.path.exists(term_path):
            os.remove(term_path)

    result = build_result(
        candidate_id,
        state,
        status=status,
        confidence=confidence,
        distance=distance,
        verified=verified,
        terminate=terminate,
        reason=reason,
        brightness=brightness_val,
    )
    timings["total_ms"] = round((time.perf_counter() - request_start) * 1000, 1)
    result["timings"] = timings
    log_check(candidate_id, result)

    print(
        f"[{result['timestamp']}] {candidate_id} -> "
        f"{status} | conf={confidence}% | brightness={brightness_val:.0f} "
        f"warns={state['warning_count']} fails={state['fail_streak']} "
        f"checks={state['check_count']}"
    )
    print(json.dumps(result, indent=2))
    return result


@app.post("/terminate_interview")
def terminate_interview(
    candidate_id: str = Form(None),
    reason: str = Form("Manual termination"),
):
    if not candidate_id:
        return api_error("Missing candidate_id")

    if candidate_id not in session_state:
        return api_error("Session not found", status_code=404)

    session_state[candidate_id]["terminated"] = True
    admin_db.delete_reference_photo(candidate_id)

    return {
        "success": True,
        "candidate_id": candidate_id,
        "message": "Interview terminated",
        "reason": reason,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/session_logs/{candidate_id}")
def session_logs(candidate_id: str):
    log_path = f"logs/{candidate_id}.json"

    if not os.path.exists(log_path):
        return {"candidate_id": candidate_id, "logs": [], "total_checks": 0}

    try:
        with open(log_path) as f:
            logs = json.load(f)
    except Exception:
        logs = []

    total = len(logs)
    verified = sum(1 for l in logs if l.get("verified") is True)
    no_face = sum(
        1
        for l in logs
        if l.get("status") in ("no_face", "side_face", "face_not_centered", "multiple_faces")
    )
    terminated = any(l.get("terminate") for l in logs)

    return {
        "candidate_id": candidate_id,
        "total_checks": total,
        "verified_checks": verified,
        "no_face_checks": no_face,
        "integrity_score": round((verified / total) * 100, 1) if total > 0 else 0,
        "session_terminated": terminated,
        "logs": logs,
    }


@app.post("/reset_session")
def reset_session(candidate_id: str = Form(None)):
    if not candidate_id:
        return api_error("Missing candidate_id")

    if candidate_id not in session_state:
        return api_error("Session not found", status_code=404)

    session_state[candidate_id] = {
        "warning_count": 0,
        "fail_streak": 0,
        "terminated": False,
        "check_count": 0,
    }

    return {
        "success": True,
        "candidate_id": candidate_id,
        "message": "Session reset",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Admin routes
# ---------------------------------------------------------------------------

def _check_admin(password: str):
    """Return True if password matches ADMIN_PASSWORD."""
    return password == ADMIN_PASSWORD


@app.post("/admin/authenticate")
def admin_authenticate(password: str = Form(...)):
    if not _check_admin(password):
        return api_error("Invalid password", status_code=401)
    return {"success": True, "message": "Authenticated"}


@app.get("/admin/candidates")
def admin_get_candidates(password: str = Query(...)):
    if not _check_admin(password):
        return api_error("Invalid password", status_code=401)

    candidates = admin_db.get_all_candidates()
    enriched = {}
    for cid, cdata in candidates.items():
        cdata["candidate_id"] = cid
        cdata["interview_summary"] = admin_db.get_interview_summary(cid)
        enriched[cid] = cdata

    return {
        "success": True,
        "candidates": enriched,
        "total_candidates": len(enriched),
    }


@app.get("/admin/candidate/{candidate_id}")
def admin_get_candidate(candidate_id: str, password: str = Query(...)):
    if not _check_admin(password):
        return api_error("Invalid password", status_code=401)

    cdata = admin_db.get_candidate(candidate_id)
    if not cdata:
        return api_error("Candidate not found", status_code=404)

    cdata["candidate_id"] = candidate_id

    return {
        "success": True,
        "candidate": cdata,
        "interview_summary": admin_db.get_interview_summary(candidate_id),
        "interview_logs": admin_db.get_interview_logs(candidate_id),
        "evidence_photos": admin_db.get_evidence_photos(candidate_id),
        "can_modify": admin_db.can_modify_candidate(candidate_id),
    }


@app.post("/admin/candidate/{candidate_id}/update")
def admin_update_candidate(
    candidate_id: str,
    password: str = Form(...),
    candidate_name: str = Form(None),
    status: str = Form(None),
    notes: str = Form(None),
):
    if not _check_admin(password):
        return api_error("Invalid password", status_code=401)

    if not admin_db.can_modify_candidate(candidate_id):
        return api_error("Cannot modify candidate within 24 hours of registration")

    updates = {}
    if candidate_name:
        updates["candidate_name"] = candidate_name
    if status:
        updates["status"] = status
    if notes is not None:
        updates["notes"] = notes

    if not admin_db.update_candidate(candidate_id, updates):
        return api_error("Candidate not found", status_code=404)

    return {"success": True, "message": "Candidate updated"}


@app.post("/admin/candidate/{candidate_id}/delete")
def admin_delete_candidate(candidate_id: str, password: str = Form(...)):
    if not _check_admin(password):
        return api_error("Invalid password", status_code=401)

    # Remove from in-memory stores too
    candidate_store.pop(candidate_id, None)
    session_state.pop(candidate_id, None)

    if not admin_db.delete_candidate(candidate_id):
        return api_error("Candidate not found", status_code=404)

    return {"success": True, "message": "Candidate deleted"}


@app.post("/admin/candidate/{candidate_id}/store-evidence")
async def admin_store_evidence(
    candidate_id: str,
    password: str = Form(...),
    file: UploadFile = File(...),
):
    if not _check_admin(password):
        return api_error("Invalid password", status_code=401)

    import tempfile
    base_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(base_dir, "evidence_temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{candidate_id}_{int(time.time())}.jpg")
    try:
        with open(temp_path, "wb") as fp:
            fp.write(await file.read())

        if not admin_db.store_evidence(candidate_id, temp_path, file.filename or None):
            return api_error("Failed to store evidence")

        return {"success": True, "message": "Evidence stored"}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/admin/cleanup/old-logs")
def admin_get_old_logs(password: str = Query(...)):
    if not _check_admin(password):
        return api_error("Invalid password", status_code=401)

    old_logs = admin_db.get_logs_older_than_24h()
    return {"success": True, "old_logs": old_logs, "count": len(old_logs)}


@app.post("/admin/cleanup/delete-old-logs")
def admin_delete_old_logs(password: str = Form(...)):
    if not _check_admin(password):
        return api_error("Invalid password", status_code=401)

    deleted_count = admin_db.delete_old_logs()
    return {
        "success": True,
        "deleted_count": deleted_count,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/admin/logs")
def admin_get_logs(password: str = Query(...)):
    if not _check_admin(password):
        return api_error("Invalid password", status_code=401)

    logs = admin_db.get_admin_logs(limit=200)
    return {"success": True, "logs": logs}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 70)
    print("  INTERVIEW.AI - FACE VERIFICATION BACKEND (FASTAPI)")
    print("=" * 70)
    print(f"  Model:        {MODEL}")
    print(f"  Detector:     {DETECTOR}")
    print(f"  Threshold:    {ARCFACE_THRESHOLD}")
    print(f"  Max Warnings: {MAX_WARNINGS}")
    print(f"  Max Fail Streak: {MAX_FAIL_STREAK}")
    print("\n  Serving: http://127.0.0.1:5000")
    print("\n" + "=" * 70 + "\n")

    uvicorn.run(app, host="127.0.0.1", port=5000)
