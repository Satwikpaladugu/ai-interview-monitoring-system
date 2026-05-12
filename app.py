"""
════════════════════════════════════════════════════════════════════
  INTERVIEW.AI — REAL-TIME FACE VERIFICATION BACKEND
════════════════════════════════════════════════════════════════════

A comprehensive Flask backend for real-time interview monitoring using
DeepFace, ArcFace embeddings, and RetinaFace detection.

Features:
  • Candidate registration with automatic reference face capture
  • Real-time identity verification via embedding comparison
  • Environment quality checks (brightness, pose, centering)
  • Warning escalation system (3 warnings → termination)
  • Session state management
  • Comprehensive audit logging
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import json
import os
import time
import tempfile
from deepface import DeepFace

# ════════════════════════════════════════════════════════════════
#  APP SETUP
# ════════════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# ════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════
MODEL             = "ArcFace"
DETECTOR          = "retinaface"
ARCFACE_THRESHOLD = 0.65

# Environment thresholds
MIN_BRIGHTNESS    = 20      # Below = too dark
MAX_BRIGHTNESS    = 242     # Above = too bright
MIN_FACE_PX       = 60      # Minimum face width/height
FACE_CENTER_RATIO = 0.12    # Margin for centering check (lenient)
EYE_GAP_RATIO     = 0.12    # Threshold for side-face detection

# Warning / termination
MAX_WARNINGS      = 3       # Total warnings before termination
MAX_FAIL_STREAK   = 3       # Consecutive identity failures before termination

# ════════════════════════════════════════════════════════════════
#  IN-MEMORY STATE
# ════════════════════════════════════════════════════════════════
candidate_store = {}  # {candidate_id: {embedding, profile_image, loaded_at, face_size}}
session_state   = {}  # {candidate_id: {warning_count, fail_streak, terminated, check_count}}


# ════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════

def cosine_distance(a, b):
    """Calculate cosine distance between two embeddings."""
    a, b = np.array(a), np.array(b)
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def calculate_confidence(distance):
    """Convert embedding distance to confidence percentage."""
    confidence = (1 - distance / ARCFACE_THRESHOLD) * 100
    return round(max(0.0, min(100.0, confidence)), 2)


def get_status(confidence):
    """Map confidence to status label."""
    if confidence >= 80:
        return "Strong Match"
    elif confidence >= 55:
        return "Possible Match"
    elif confidence >= 40:
        return "Weak Match"
    else:
        return "No Match"


def log_check(candidate_id, result):
    """Append verification result to JSON log file."""
    log_path = f"logs/{candidate_id}.json"
    logs = []
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                logs = json.load(f)
        except:
            logs = []
    logs.append(result)
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=2)


def check_brightness(img):
    """
    Check image brightness and return status.
    Returns (ok: bool, brightness: float, label: str)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    
    if brightness < MIN_BRIGHTNESS:
        return False, brightness, "too_dark"
    if brightness > MAX_BRIGHTNESS:
        return False, brightness, "too_bright"
    return True, brightness, "ok"


def check_face_pose(frame_path):
    """
    Detect face and validate confidence.
    Returns (face_found: bool, results: list)
    """
    try:
        results = DeepFace.represent(
            img_path=frame_path,
            model_name=MODEL,
            detector_backend=DETECTOR,
            enforce_detection=True
        )
        if not results or len(results) == 0:
            return False, []
        
        # Filter very low confidence detections
        valid = [r for r in results if r.get("face_confidence", 1.0) >= 0.70]
        if not valid:
            return False, []
        return True, valid
    except Exception as e:
        print(f"[check_face_pose] Exception: {e}")
        return False, []


def build_result(candidate_id, state, status, confidence, distance,
                 verified, terminate, reason, brightness=None):
    """Build standardized API response."""
    r = {
        "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_id":  candidate_id,
        "status":        status,
        "confidence":    confidence,
        "distance":      round(distance, 4) if distance is not None else None,
        "verified":      verified,
        "warning_count": state["warning_count"],
        "fail_streak":   state["fail_streak"],
        "check_count":   state["check_count"],
        "terminate":     terminate,
        "reason":        reason,
        "brightness":    round(brightness, 1) if brightness is not None else None,
    }
    return r


# ════════════════════════════════════════════════════════════════
#  ROUTE 1 — Health Check
# ════════════════════════════════════════════════════════════════
@app.route('/health', methods=['GET'])
def health():
    """Return backend health status and configuration."""
    return jsonify({
        "status":                "running",
        "model":                 MODEL,
        "detector":              DETECTOR,
        "threshold":             ARCFACE_THRESHOLD,
        "registered_candidates": len(candidate_store),
        "timestamp":             time.strftime("%Y-%m-%d %H:%M:%S")
    })


# ════════════════════════════════════════════════════════════════
#  ROUTE 2 — Start Interview (Register Candidate & Generate Embedding)
# ════════════════════════════════════════════════════════════════
@app.route('/start_interview', methods=['POST'])
def start_interview():
    """
    Accept candidate name and reference image.
    Generate and store face embedding.
    Initialize session state.
    """
    candidate_id = request.form.get('candidate_id')
    candidate_name = request.form.get('candidate_name')
    
    if not candidate_id or not candidate_name:
        return jsonify({"error": "Missing candidate_id or candidate_name"}), 400
    
    if 'file' not in request.files:
        return jsonify({"error": "No reference image provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    # Load reference image into temporary file (TEMPORARY - deleted immediately after processing)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)
    
    try:
        print("=" * 50)
        print("TEMP PROFILE PATH:", tmp_path)

        img = cv2.imread(tmp_path)

        print("IMG:", img)
        print("IMG SHAPE:", img.shape if img is not None else None)
        print("=" * 50)
        
        # Generate embedding from reference image
        try:
            results = DeepFace.represent(
                img_path=tmp_path,
                model_name=MODEL,
                detector_backend=DETECTOR,
                enforce_detection=True
            )
        except ValueError as e:
            return jsonify({"error": "No face found in reference image"}), 400
        except Exception as e:
            return jsonify({"error": f"Failed to process reference image: {str(e)}"}), 400
        
        if len(results) == 0:
            return jsonify({"error": "No face detected in reference image"}), 400
        
        if len(results) > 1:
            return jsonify({"error": "Multiple faces in reference image"}), 400
        
        # Validate face size
        facial_area = results[0].get("facial_area", {})
        face_w = facial_area.get("w", 0)
        face_h = facial_area.get("h", 0)
        
        if face_w < 80 or face_h < 80:
            return jsonify({"error": "Face in reference image is too small"}), 400
        
        # Store candidate (ONLY embedding in memory, reference image file is temporary)
        candidate_store[candidate_id] = {
            "embedding":     results[0]["embedding"],
            "candidate_name": candidate_name,
            "loaded_at":     time.strftime("%Y-%m-%d %H:%M:%S"),
            "face_size":     f"{face_w}x{face_h}px"
        }
    finally:
        # Delete temporary file immediately after processing
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                print(f"[CLEANUP] Deleted temporary reference image: {tmp_path}")
            except Exception as e:
                print(f"[ERROR] Failed to delete temp file: {e}")
    
    # Initialize session state
    session_state[candidate_id] = {
        "warning_count": 0,
        "fail_streak":   0,
        "terminated":    False,
        "check_count":   0
    }
    
    return jsonify({
        "success":       True,
        "candidate_id":  candidate_id,
        "candidate_name": candidate_name,
        "message":       "Interview started. Monitoring enabled.",
        "face_size":     f"{face_w}x{face_h}px"
    })


# ════════════════════════════════════════════════════════════════
#  ROUTE 3 — Verify Frame (Main Monitoring Endpoint)
# ════════════════════════════════════════════════════════════════
@app.route('/verify_frame', methods=['POST'])
def verify_frame():
    """
    Main verification endpoint called during interview.
    Performs all checks: brightness, face pose, identity verification.
    Manages warning escalation and termination.
    """
    candidate_id = request.form.get('candidate_id')
    
    if not candidate_id:
        return jsonify({"error": "Missing candidate_id"}), 400
    
    if candidate_id not in candidate_store:
        return jsonify({"error": "Candidate not registered"}), 400
    
    if 'file' not in request.files:
        return jsonify({"error": "No frame provided"}), 400
    
    # Ensure session state exists
    if candidate_id not in session_state:
        session_state[candidate_id] = {
            "warning_count": 0,
            "fail_streak":   0,
            "terminated":    False,
            "check_count":   0
        }
    
    state = session_state[candidate_id]
    
    # Check if already terminated
    if state["terminated"]:
        return jsonify({
            "terminate":   True,
            "reason":      "Session already terminated",
            "status":      "terminated",
            "verified":    False,
            "confidence":  0,
            "warning_count": state["warning_count"],
            "fail_streak": state["fail_streak"],
            "check_count": state["check_count"],
            "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # ──────────────────────────────────────────────────────────
    # SAVE FRAME
    # ──────────────────────────────────────────────────────────
    timestamp = int(time.time())
    frame_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        f"frame_{candidate_id}_{timestamp}.jpg"
    )
    request.files['file'].save(frame_path)
    state["check_count"] += 1
    
    def cleanup():
        """Remove temporary frame file."""
        if os.path.exists(frame_path):
            try:
                os.remove(frame_path)
            except:
                pass
    
    # ──────────────────────────────────────────────────────────
    # LOAD AND VALIDATE IMAGE
    # ──────────────────────────────────────────────────────────
    img = cv2.imread(frame_path)
    if img is None:
        cleanup()
        return jsonify({"error": "Failed to read frame"}), 400
    
    frame_h, frame_w = img.shape[:2]
    terminate = False
    reason = None
    
    # ──────────────────────────────────────────────────────────
    # CHECK A: BRIGHTNESS
    # ──────────────────────────────────────────────────────────
    bright_ok, brightness_val, bright_label = check_brightness(img)
    
    if not bright_ok:
        state["warning_count"] += 1
        warn_label = "too_dark" if bright_label == "too_dark" else "too_bright"
        status = warn_label
        confidence = 0
        verified = False
        distance = None
        
        result = build_result(
            candidate_id, state,
            status=status, confidence=confidence,
            distance=distance, verified=verified,
            terminate=terminate, reason=reason,
            brightness=brightness_val
        )
        log_check(candidate_id, result)
        cleanup()
        
        # Check termination after warning
        if state["warning_count"] >= MAX_WARNINGS:
            state["terminated"] = True
            result["terminate"] = True
            result["reason"] = "Too many warnings"
            log_check(candidate_id, result)
        
        result["suggestions"] = [
            "Turn on more lights" if bright_label == "too_dark" else "Reduce glare/brightness"
        ]
        print(json.dumps(result, indent=2))
        return jsonify(result)
    
    # ──────────────────────────────────────────────────────────
    # CHECK B: FACE DETECTION & POSITIONING
    # ──────────────────────────────────────────────────────────
    face_found, results = check_face_pose(frame_path)
    
    if not face_found or len(results) == 0:
        state["warning_count"] += 1
        status = "no_face"
        confidence = 0
        verified = False
        distance = None
        
        result = build_result(
            candidate_id, state,
            status=status, confidence=confidence,
            distance=distance, verified=verified,
            terminate=terminate, reason=reason,
            brightness=brightness_val
        )
        log_check(candidate_id, result)
        cleanup()
        
        if state["warning_count"] >= MAX_WARNINGS:
            state["terminated"] = True
            result["terminate"] = True
            result["reason"] = "Too many warnings"
            log_check(candidate_id, result)
        
        result["suggestions"] = ["Ensure your face is visible to the camera"]
        print(json.dumps(result, indent=2))
        return jsonify(result)
    
    # ──────────────────────────────────────────────────────────
    # CHECK B2: MULTIPLE PERSONS DETECTED
    # ──────────────────────────────────────────────────────────
    if len(results) > 1:
        state["warning_count"] += 1
        state["fail_streak"] += 1
        status = "multiple_faces"
        confidence = 0
        verified = False
        distance = None
        reason_multiple = f"Multiple persons detected ({len(results)} faces)"
        
        terminate_now = (
            state["fail_streak"] >= MAX_FAIL_STREAK or
            state["warning_count"] >= MAX_WARNINGS
        )
        if terminate_now:
            state["terminated"] = True
        
        result = build_result(
            candidate_id, state,
            status=status, confidence=confidence,
            distance=distance, verified=verified,
            terminate=terminate_now,
            reason=reason_multiple,
            brightness=brightness_val
        )
        log_check(candidate_id, result)
        cleanup()
        
        result["suggestions"] = ["Only one person should be visible to the camera"]
        print(f"[MULTIPLE FACES] {reason_multiple}")
        print(json.dumps(result, indent=2))
        return jsonify(result)
    
    # ──────────────────────────────────────────────────────────
    # CHECK C: FACE SIZE & CENTERING
    # ──────────────────────────────────────────────────────────
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
        status = "face_not_centered"
        confidence = 0
        verified = False
        distance = None
        
        result = build_result(
            candidate_id, state,
            status=status, confidence=confidence,
            distance=distance, verified=verified,
            terminate=terminate, reason=reason,
            brightness=brightness_val
        )
        log_check(candidate_id, result)
        cleanup()
        
        if state["warning_count"] >= MAX_WARNINGS:
            state["terminated"] = True
            result["terminate"] = True
            result["reason"] = "Too many warnings"
            log_check(candidate_id, result)
        
        suggestions = []
        if too_small:
            suggestions.append("Move closer to camera")
        if too_left or too_right:
            suggestions.append("Center your face horizontally")
        if too_up or too_down:
            suggestions.append("Center your face vertically")
        result["suggestions"] = suggestions
        print(json.dumps(result, indent=2))
        return jsonify(result)
    
    # ──────────────────────────────────────────────────────────
    # CHECK D: SIDE FACE / LOOKING AWAY
    # ──────────────────────────────────────────────────────────
    landmarks = results[0].get("facial_area", {})
    left_eye = landmarks.get("left_eye")
    right_eye = landmarks.get("right_eye")
    
    side_face = False
    if left_eye and right_eye and face_w > 0:
        eye_gap = abs(right_eye[0] - left_eye[0])
        if eye_gap < EYE_GAP_RATIO * face_w:
            side_face = True
    
    if side_face:
        state["warning_count"] += 1
        status = "side_face"
        confidence = 0
        verified = False
        distance = None
        
        result = build_result(
            candidate_id, state,
            status=status, confidence=confidence,
            distance=distance, verified=verified,
            terminate=terminate, reason=reason,
            brightness=brightness_val
        )
        log_check(candidate_id, result)
        cleanup()
        
        if state["warning_count"] >= MAX_WARNINGS:
            state["terminated"] = True
            result["terminate"] = True
            result["reason"] = "Too many warnings"
            log_check(candidate_id, result)
        
        result["suggestions"] = ["Look directly at the camera"]
        print(json.dumps(result, indent=2))
        return jsonify(result)
    
    # ──────────────────────────────────────────────────────────
    # CHECK E: IDENTITY VERIFICATION
    # ──────────────────────────────────────────────────────────
    webcam_embedding = results[0]["embedding"]
    stored_embedding = candidate_store[candidate_id]["embedding"]
    
    distance = cosine_distance(stored_embedding, webcam_embedding)
    confidence = calculate_confidence(distance)
    status = get_status(confidence)
    verified = distance < ARCFACE_THRESHOLD
    
    # Update fail streak
    if verified and confidence >= 75:
        state["fail_streak"] = 0

    elif verified and confidence >= 45:
        state["fail_streak"] = max(0, state["fail_streak"] - 1)

    else:
        state["fail_streak"] += 1
    
    # ──────────────────────────────────────────────────────────
    # CHECK TERMINATION CONDITIONS
    # ──────────────────────────────────────────────────────────
    if state["fail_streak"] >= MAX_FAIL_STREAK:
        state["terminated"] = True
        terminate = True
        reason = "Identity verification failed repeatedly"
    
    if state["warning_count"] >= MAX_WARNINGS:
        state["terminated"] = True
        terminate = True
        reason = "Too many warnings"
    
    # ──────────────────────────────────────────────────────────
    # BUILD AND RETURN RESULT
    # ──────────────────────────────────────────────────────────
    result = build_result(
        candidate_id, state,
        status=status, confidence=confidence,
        distance=distance, verified=verified,
        terminate=terminate, reason=reason,
        brightness=brightness_val
    )
    log_check(candidate_id, result)
    cleanup()
    
    print(
        f"[{result['timestamp']}] {candidate_id} → "
        f"{status} | conf={confidence}% | brightness={brightness_val:.0f} "
        f"warns={state['warning_count']} fails={state['fail_streak']} "
        f"checks={state['check_count']}"
    )
    print(json.dumps(result, indent=2))
    return jsonify(result)


# ════════════════════════════════════════════════════════════════
#  ROUTE 4 — Terminate Interview
# ════════════════════════════════════════════════════════════════
@app.route('/terminate_interview', methods=['POST'])
def terminate_interview():
    """Manually terminate an interview session."""
    candidate_id = request.form.get('candidate_id')
    reason = request.form.get('reason', 'Manual termination')
    
    if not candidate_id:
        return jsonify({"error": "Missing candidate_id"}), 400
    
    if candidate_id not in session_state:
        return jsonify({"error": "Session not found"}), 404
    
    session_state[candidate_id]["terminated"] = True
    
    # Remove candidate from store (embedding was only in memory)
    if candidate_id in candidate_store:
        del candidate_store[candidate_id]
        print(f"[CLEANUP] Removed candidate {candidate_id} from memory")
    
    return jsonify({
        "success":       True,
        "candidate_id":  candidate_id,
        "message":       "Interview terminated",
        "reason":        reason,
        "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S")
    })


# ════════════════════════════════════════════════════════════════
#  ROUTE 5 — Session Logs
# ════════════════════════════════════════════════════════════════
@app.route('/session_logs/<candidate_id>', methods=['GET'])
def session_logs(candidate_id):
    """Retrieve complete session audit log."""
    log_path = f"logs/{candidate_id}.json"
    
    if not os.path.exists(log_path):
        return jsonify({
            "candidate_id": candidate_id,
            "logs": [],
            "total_checks": 0
        })
    
    try:
        with open(log_path) as f:
            logs = json.load(f)
    except:
        logs = []
    
    total = len(logs)
    verified = sum(1 for l in logs if l.get('verified') is True)
    no_face = sum(1 for l in logs if l.get('status') in 
                  ('no_face', 'side_face', 'face_not_centered'))
    terminated = any(l.get('terminate') for l in logs)
    
    return jsonify({
        "candidate_id":    candidate_id,
        "total_checks":    total,
        "verified_checks": verified,
        "no_face_checks":  no_face,
        "integrity_score": round((verified / total) * 100, 1) if total > 0 else 0,
        "session_terminated": terminated,
        "logs":             logs
    })


# ════════════════════════════════════════════════════════════════
#  ROUTE 6 — Reset Session
# ════════════════════════════════════════════════════════════════
@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Reset session state for a candidate."""
    candidate_id = request.form.get('candidate_id')
    
    if not candidate_id:
        return jsonify({"error": "Missing candidate_id"}), 400
    
    if candidate_id not in session_state:
        return jsonify({"error": "Session not found"}), 404
    
    # Remove candidate from store (embedding was only in memory)
    if candidate_id in candidate_store:
        del candidate_store[candidate_id]
        print(f"[CLEANUP] Removed candidate {candidate_id} from memory")
    
    session_state[candidate_id] = {
        "warning_count": 0,
        "fail_streak":   0,
        "terminated":    False,
        "check_count":   0
    }
    
    return jsonify({
        "success":       True,
        "candidate_id":  candidate_id,
        "message":       "Session reset",
        "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S")
    })


# ════════════════════════════════════════════════════════════════
#  STARTUP
# ════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "="*70)
    print("  🎬 INTERVIEW.AI — FACE VERIFICATION BACKEND")
    print("="*70)
    print(f"  Model:        {MODEL}")
    print(f"  Detector:     {DETECTOR}")
    print(f"  Threshold:    {ARCFACE_THRESHOLD}")
    print(f"  Brightness:   {MIN_BRIGHTNESS}–{MAX_BRIGHTNESS}")
    print(f"  Max Warnings: {MAX_WARNINGS}")
    print(f"  Max Fail Streak: {MAX_FAIL_STREAK}")
    print("\n  Endpoints:")
    print("    GET  /health              — Backend status")
    print("    POST /start_interview     — Register candidate & capture reference")
    print("    POST /verify_frame        — Verify webcam frame during interview")
    print("    POST /terminate_interview — Manually end interview")
    print("    GET  /session_logs/<id>   — Retrieve session audit log")
    print("    POST /reset_session       — Reset session state")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, port=5000, host='127.0.0.1')
