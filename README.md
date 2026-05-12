# 🎬 interview.ai — Real-Time AI Interview Monitoring System

A comprehensive Flask-based system for real-time interview monitoring using facial recognition, embeddings-based identity verification, and environmental quality detection.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Interview Flow](#interview-flow)
- [Detection Features](#detection-features)
- [Warning & Termination System](#warning--termination-system)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)

## ✨ Features

### Core Capabilities

✅ **Real-Time Facial Recognition**
- ArcFace embeddings for identity verification
- Cosine similarity comparison
- 68% threshold for confident matches

✅ **Environmental Quality Checks**
- Brightness analysis (20–242 range, 0–255 scale)
- Face pose detection (side-face, centered)
- Face size validation (min 60×60 pixels)
- Frame centering validation

✅ **Interview Flow Management**
- Candidate registration with reference photo capture
- Automatic embedding generation from reference
- Per-session state management
- Warning escalation system (3 warnings → termination)
- Consecutive identity failure tracking

✅ **Modern Frontend UI**
- Dark-themed responsive design
- Live webcam preview with status badges
- Real-time counters and progress bars
- Comprehensive monitoring logs
- Full-screen termination overlay

✅ **Session Management & Logging**
- Per-candidate JSON audit logs
- Complete verification history
- Session state persistence (in-memory)
- Timestamp tracking for all events

## 🏗️ Architecture

### Backend Stack
- **Framework:** Flask + CORS
- **Face Recognition:** DeepFace (ArcFace model)
- **Face Detection:** RetinaFace
- **Image Processing:** OpenCV + NumPy
- **Port:** 127.0.0.1:5000

### Frontend Stack
- **HTML5:** Video elements, canvas, forms
- **CSS3:** Modern dark theme, animations, responsive grid
- **JavaScript:** Async/await, FormData API, WebRTC
- **Intervals:** 30-second verification loop

### Data Flow

```
┌─────────────┐
│   Browser   │ (HTML5 Webcam API)
└──────┬──────┘
       │
       ├─► POST /start_interview (reference image)
       │   ↓
       │   [Generate Embedding & Store]
       │   ↓ (success)
       │
       ├─► POST /verify_frame (every 30 sec)
       │   ↓
       │   [Brightness Check]
       │   ├─► [Face Detection]
       │   │   ├─► [Pose Validation]
       │   │   └─► [Side-Face Check]
       │   └─► [Identity Verification]
       │   ↓
       │   [Check Termination Conditions]
       │   ↓
       │   JSON Response
       │
       └─► UI Update & Logs
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Pip package manager
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Webcam access

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Start the Backend

```bash
python app.py
```

Expected output:
```
======================================================================
  🎬 INTERVIEW.AI — FACE VERIFICATION BACKEND
======================================================================
  Model:        ArcFace
  Detector:     RetinaFace
  Threshold:    0.68
  Brightness:   20–242
  Max Warnings: 3
  Max Fail Streak: 3

  Endpoints:
    GET  /health              — Backend status
    POST /start_interview     — Register candidate & capture reference
    POST /verify_frame        — Verify webcam frame during interview
    POST /terminate_interview — Manually end interview
    GET  /session_logs/<id>   — Retrieve session audit log
    POST /reset_session       — Reset session state

======================================================================
```

### Step 3: Open Frontend

Open in your browser:
```
file:///c:/Users/satwi/OneDrive/Desktop/image detection/index.html
```

Or serve with Python:
```bash
python -m http.server 8000
# Then visit: http://localhost:8000/index.html
```

## ⚙️ Configuration

Edit these constants in `app.py` to customize behavior:

```python
# Model & Detection
MODEL             = "ArcFace"          # Embedding model
DETECTOR          = "retinaface"       # Face detector
ARCFACE_THRESHOLD = 0.68               # Confidence threshold

# Environment Checks
MIN_BRIGHTNESS    = 20                 # Too dark threshold
MAX_BRIGHTNESS    = 242                # Too bright threshold
MIN_FACE_PX       = 60                 # Minimum face size
FACE_CENTER_RATIO = 0.12               # Center margin (lenient)
EYE_GAP_RATIO     = 0.12               # Side-face detection sensitivity

# Warning/Termination
MAX_WARNINGS      = 3                  # Total warnings before termination
MAX_FAIL_STREAK   = 3                  # Consecutive failures before termination
```

Frontend config in `index.html` (JavaScript):
```javascript
const API = "http://127.0.0.1:5000";    // Backend URL
const VERIFY_INTERVAL = 30000;          // Check interval (ms)
```

## 📡 API Documentation

### 1. Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "running",
  "model": "ArcFace",
  "detector": "retinaface",
  "threshold": 0.68,
  "registered_candidates": 5,
  "timestamp": "2024-05-08 14:30:45"
}
```

---

### 2. Start Interview

**Endpoint:** `POST /start_interview`

**Request:**
```
Form Data:
  - candidate_id (string): Unique candidate identifier
  - candidate_name (string): Candidate's full name
  - file (image/jpeg): Reference photo (from webcam)
```

**Response (Success):**
```json
{
  "success": true,
  "candidate_id": "candidate_1715168445000",
  "candidate_name": "John Doe",
  "message": "Interview started. Monitoring enabled.",
  "face_size": "320x240px"
}
```

**Response (Error):**
```json
{
  "error": "No face found in reference image"
}
```

**Errors:**
- "No face found in reference image" — Face detection failed
- "Multiple faces in reference image" — Ambiguous reference
- "Face in reference image is too small" — Face < 80×80 px

---

### 3. Verify Frame

**Endpoint:** `POST /verify_frame`

**Request:**
```
Form Data:
  - candidate_id (string): Candidate identifier
  - file (image/jpeg): Current webcam frame
```

**Response (Normal):**
```json
{
  "timestamp": "2024-05-08 14:35:22",
  "candidate_id": "candidate_1715168445000",
  "status": "Strong Match",
  "confidence": 92.5,
  "distance": 0.15,
  "verified": true,
  "warning_count": 1,
  "fail_streak": 0,
  "check_count": 5,
  "terminate": false,
  "reason": null,
  "brightness": 128.5
}
```

**Response (Termination):**
```json
{
  "timestamp": "2024-05-08 14:36:01",
  "candidate_id": "candidate_1715168445000",
  "status": "No Match",
  "confidence": 25.0,
  "distance": 1.42,
  "verified": false,
  "warning_count": 3,
  "fail_streak": 3,
  "check_count": 12,
  "terminate": true,
  "reason": "Identity verification failed repeatedly",
  "brightness": 145.0
}
```

**Status Values:**
- `Strong Match` — Identity verified (confidence ≥ 80%)
- `Possible Match` — Likely match (55% ≤ conf < 80%)
- `Weak Match` — Uncertain (40% ≤ conf < 55%)
- `No Match` — Different person (conf < 40%)
- `no_face` — No face detected
- `side_face` — Looking away (eye gap too small)
- `face_not_centered` — Face outside center region
- `too_dark` — Brightness < 20
- `too_bright` — Brightness > 242

---

### 4. Terminate Interview

**Endpoint:** `POST /terminate_interview`

**Request:**
```
Form Data:
  - candidate_id (string): Candidate identifier
  - reason (string, optional): Termination reason
```

**Response:**
```json
{
  "success": true,
  "candidate_id": "candidate_1715168445000",
  "message": "Interview terminated",
  "reason": "Manual termination",
  "timestamp": "2024-05-08 14:37:15"
}
```

---

### 5. Session Logs

**Endpoint:** `GET /session_logs/<candidate_id>`

**Response:**
```json
{
  "candidate_id": "candidate_1715168445000",
  "total_checks": 12,
  "verified_checks": 10,
  "no_face_checks": 2,
  "integrity_score": 83.3,
  "session_terminated": false,
  "logs": [
    {
      "timestamp": "2024-05-08 14:35:10",
      "candidate_id": "candidate_1715168445000",
      "status": "Strong Match",
      "confidence": 89.5,
      "distance": 0.19,
      "verified": true,
      "warning_count": 0,
      "fail_streak": 0,
      "check_count": 1,
      "terminate": false,
      "reason": null,
      "brightness": 130.2
    },
    ...
  ]
}
```

---

### 6. Reset Session

**Endpoint:** `POST /reset_session`

**Request:**
```
Form Data:
  - candidate_id (string): Candidate identifier
```

**Response:**
```json
{
  "success": true,
  "candidate_id": "candidate_1715168445000",
  "message": "Session reset",
  "timestamp": "2024-05-08 14:38:00"
}
```

## 🎯 Interview Flow

### Phase 1: Welcome
1. Candidate enters full name
2. Frontend validates input
3. System generates unique candidate ID

### Phase 2: Reference Capture
1. Browser requests webcam access
2. Live preview shows in reference panel
3. Candidate clicks "Capture Reference Photo"
4. Frame captured and prepared as FormData
5. Backend receives and processes reference image:
   - Detects face using RetinaFace
   - Validates single face, size > 80×80 px
   - Generates ArcFace embedding
   - Stores embedding + metadata
6. Session state initialized (warnings=0, fails=0)
7. Candidate sees "Start Interview" button with guidelines

### Phase 3: Monitoring Loop
1. Interview begins when clicking "Start Interview"
2. Every 30 seconds:
   - Frontend captures webcam frame via canvas
   - Measures brightness on client side
   - Sends JPEG blob to `/verify_frame`
3. Backend performs 5-step verification:
   - **Check A:** Brightness validation
   - **Check B:** Face detection + confidence
   - **Check C:** Face size & centering
   - **Check D:** Side-face detection (eye gap)
   - **Check E:** Identity verification (cosine distance)
4. Returns JSON with status, counters, suggestions
5. Frontend updates UI:
   - Confidence bar updates
   - Counters increment
   - Status text changes
   - Logs grow
   - Video border color changes (green/yellow/red)

### Phase 4: Termination (if triggered)
1. Backend detects termination condition:
   - `warning_count >= 3` OR
   - `fail_streak >= 3`
2. Sets `terminate: true` in response
3. Frontend receives and shows red overlay
4. Monitoring loop stops
5. Candidate sees termination reason
6. "Start New Session" button reloads page

## 🚨 Detection Features

### 1. Brightness Detection

**Logic:**
- Convert frame to grayscale
- Calculate mean pixel intensity (0–255)
- Flag if `brightness < 20` (too_dark) or `brightness > 242` (too_bright)

**Triggers Warning:** Yes

**Suggestions:**
- Too Dark: "Turn on more lights"
- Too Bright: "Reduce glare/brightness"

---

### 2. Face Detection

**Logic:**
- Use RetinaFace detector
- Set `enforce_detection=False` (lenient) to avoid crashes
- Filter results by confidence >= 0.70
- Require non-empty result set

**Triggers Warning:** Yes (no_face status)

**Suggestions:**
- "Ensure your face is visible to the camera"

---

### 3. Face Size Check

**Logic:**
- Get facial bounding box from detector
- Validate width and height >= 60px
- Prevents false positives from tiny faces

**Triggers Warning:** Yes (face_not_centered status)

**Suggestions:**
- "Move closer to camera"

---

### 4. Face Centering

**Logic:**
- Calculate face center: `(box_x + box_w/2, box_y + box_h/2)`
- Check against frame dimensions
- Lenient margins: only flag if center in outer 12%:
  - `center_x < frame_w * 0.12` (too far left)
  - `center_x > frame_w * 0.88` (too far right)
  - Same for vertical

**Triggers Warning:** Yes (face_not_centered status)

**Suggestions:**
- "Center your face horizontally" / vertically

---

### 5. Side-Face / Looking Away

**Logic:**
- Extract eye landmarks: `left_eye`, `right_eye`
- Calculate horizontal eye gap: `|right_eye_x - left_eye_x|`
- Flag if gap < 12% of face width (extreme turn)

**Triggers Warning:** Yes (side_face status)

**Suggestions:**
- "Look directly at the camera"

---

### 6. Identity Verification

**Logic:**
- Get ArcFace embedding from current frame
- Compare with stored reference embedding
- Calculate cosine distance
- Convert to confidence: `(1 - distance/0.68) * 100`
- Classify by thresholds:
  - `confidence >= 80`: Strong Match ✅
  - `55 <= confidence < 80`: Possible Match 🟡
  - `40 <= confidence < 55`: Weak Match 🟠
  - `confidence < 40`: No Match ❌
- `verified = (distance < 0.68)`

**Tracks Fail Streak:**
- Increments if `!verified`
- Resets to 0 if `verified`
- Termination if `fail_streak >= 3`

**Does NOT directly trigger warning** (separate logic)

## ⚠️ Warning & Termination System

### Warning System

**What is a warning?**
- One point for each violation detected in a verification check
- Violations: brightness, no face, bad centering, side face, etc.
- **NOT counting** identity mismatch as direct warning

**Escalation:**
- Warning 1: Yellow status, continues monitoring
- Warning 2: Detailed suggestions shown
- Warning 3: **SESSION TERMINATED** 🚫

**Reset:** Warnings never reset during a session (accumulate)

### Fail Streak System

**What is a fail streak?**
- Consecutive frames where identity verification failed
- Counter increments if `!verified` (confidence < 68%)
- Counter resets to 0 if `verified`

**Escalation:**
- Fail 1–2: Orange/red status, suggestions shown
- Fail 3: **SESSION TERMINATED** 🚫

**Logic:** Distinct from warnings. Candidate can have 1–2 fail streaks and recover if next check succeeds.

### Termination Triggers

Interview **TERMINATES** if either condition is true:
```
state["warning_count"] >= MAX_WARNINGS (3)     OR
state["fail_streak"] >= MAX_FAIL_STREAK (3)
```

**On Termination:**
1. Backend sets `terminate: true` in response
2. Sets `terminated: true` in session state
3. Logs termination event with reason
4. Frontend:
   - Clears monitoring timers
   - Shows full-screen red overlay
   - Displays termination reason
   - Disables monitoring loop
   - Shows "Start New Session" button

**Post-Termination:**
- All further `/verify_frame` calls for that candidate return:
  ```json
  {
    "terminate": true,
    "reason": "Session already terminated",
    "verified": false,
    "confidence": 0
  }
  ```
- Candidate must reload to start new session

## 📁 File Structure

```
image detection/
├── app.py                    # Flask backend (Main server)
├── index.html                # Frontend HTML + CSS + JS (All-in-one)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── uploads/                  # Uploaded files
│   ├── reference_*.jpg       # Reference photos
│   └── frame_*.jpg           # Temporary verification frames
│
├── logs/                     # Session audit logs
│   ├── candidate_*.json      # Per-candidate verification history
│   └── ...
│
├── app_old.py                # Previous backend (backup)
└── index_old.html            # Previous frontend (backup)
```

### File Details

**app.py** (800+ lines)
- Flask app initialization
- 6 API endpoints with full error handling
- Helper functions (embedding, brightness, pose checking)
- State management (in-memory dicts)
- Comprehensive logging

**index.html** (900+ lines)
- Embedded CSS (dark theme, animations, responsive)
- Three-phase interview flow UI
- JavaScript monitoring loop with canvas API
- Real-time status updates
- Webcam video element

**requirements.txt**
- Flask 2.3.2
- Flask-CORS 4.0.0
- DeepFace 0.0.75
- OpenCV 4.8.0.74
- NumPy 1.24.3
- TensorFlow 2.12.0

## 🛠️ Troubleshooting

### Problem: "Backend Offline" in health pill

**Cause:** Flask server not running or different port.

**Solution:**
```bash
# Check if Flask is running
python app.py

# Verify port 5000 is accessible
# Check firewall settings
```

---

### Problem: "Camera access denied"

**Cause:** Browser permission not granted or HTTPS required.

**Solution:**
1. Allow camera access when browser asks
2. For HTTPS, configure Flask with SSL
3. Check browser console for specific error

---

### Problem: "No face found in reference image"

**Cause:** RetinaFace couldn't detect a face.

**Reasons:**
- Bad lighting
- Face too small or rotated
- Multiple faces
- Non-face image

**Solution:**
- Use good lighting
- Face the camera directly
- Ensure face is large enough
- Only one person in frame

---

### Problem: "Multiple faces in reference image"

**Cause:** More than one person detected.

**Solution:**
- Ensure only candidate is in frame
- Remove other people from background

---

### Problem: Reference photo too small

**Cause:** Face detected but < 80×80 pixels.

**Solution:**
- Move closer to camera
- Improve distance/focal length
- Increase zoom/resolution

---

### Problem: Monitoring stops after a few checks

**Cause:** 
- MediaStream ended (camera disconnected)
- JavaScript error
- Browser tab inactive

**Solution:**
1. Check browser console (F12 → Console)
2. Reconnect camera
3. Keep tab in focus
4. Reload page if stuck

---

### Problem: "Confidence never above 50%"

**Cause:** Reference and live frames too different.

**Reasons:**
- Lighting change
- Different camera angle/distance
- Appearance change (glasses, facial hair)
- Reference image quality poor

**Solution:**
- Retake reference in similar lighting
- Maintain consistent distance/angle during interview
- Use good lighting throughout

---

### Problem: Always getting "Too Dark" / "Too Bright"

**Cause:** Brightness threshold misconfigured.

**Solution:**
Edit in `app.py`:
```python
MIN_BRIGHTNESS = 15   # Lower if too sensitive
MAX_BRIGHTNESS = 245  # Higher if too sensitive
```

---

### Problem: Session logs not saving

**Cause:** Permissions issue on `logs/` directory.

**Solution:**
```bash
# Ensure logs directory is writable
chmod 755 logs/

# Or verify permissions in Windows
# Ensure user has write access to folder
```

---

### Problem: Slow verification / timeout

**Cause:** DeepFace embedding generation is slow on first run.

**Solution:**
1. First time uses downloads: first run (~2GB download)
2. Subsequent runs faster (cached)
3. Consider GPU acceleration:
   ```python
   # Add to app.py imports
   os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0
   ```

---

### Problem: CORS errors in browser console

**Cause:** Flask-CORS not allowing requests.

**Solution:**
1. Verify `Flask-CORS` is installed
2. Check URL in index.html:
   ```javascript
   const API = "http://127.0.0.1:5000";
   ```
3. Ensure Flask is actually running on that address/port

---

### Problem: "Session already terminated" after restart

**Cause:** Session state persisted from previous interview.

**Solution:**
1. Call `/reset_session` endpoint
2. Or reload page to get new `candidate_id`
3. State is in-memory, lost on server restart

---

## 📊 Example Interview Session

```
Timeline: 0 sec → Interview Starts
- Candidate: John Doe
- ID: candidate_1715168445000
- Reference: captured from webcam

Timeline: 30 sec → First Check
✓ Brightness: 125 (OK)
✓ Face detected, confidence 0.92
✓ Face 280x210px (OK)
✓ Centered properly
✓ Looking at camera
✓ Identity match: 89.5% confidence ✅
Result: Strong Match, 0 warnings, 0 fails

Timeline: 60 sec → Second Check
✓ Brightness: 110 (OK)
⚠ Face detected but only 240x180px
⚠ Right edge close to frame edge
✓ Looking at camera
✓ Identity match: 84.2% confidence ✅
Result: Strong Match, 1 warning, 0 fails

Timeline: 90 sec → Third Check
✗ Brightness: 15 (TOO DARK) ❌
→ Warning count: 2 (1 + brightness warning)
Result: Too Dark, 2 warnings, triggers suggestions

Timeline: 120 sec → Fourth Check
✓ Brightness: 130 (OK after lights turned on)
✗ Face not detected ❌
→ Warning count: 3 (2 + no face warning)
→ INTERVIEW TERMINATED 🛑

Reason: Too many warnings
Final Stats:
- Total checks: 4
- Successful verifications: 2
- No-face checks: 1
- Warnings accumulated: 3
- Interview duration: 120 seconds
```

## 📞 Support & Next Steps

### Customization Ideas

1. **Increase Verification Interval:**
   ```javascript
   const VERIFY_INTERVAL = 60000;  // 60 seconds
   ```

2. **Stricter Identity Threshold:**
   ```python
   ARCFACE_THRESHOLD = 0.60  # More strict
   ```

3. **More Lenient Warnings:**
   ```python
   MAX_WARNINGS = 5  # Allow more violations
   ```

4. **Add Eye Contact Detection:**
   - Track gaze using face landmarks
   - Flag if looking away > 50% of time

5. **Add Sound Alerts:**
   - Warning beep on violation
   - Success chime on good verification

6. **Database Integration:**
   - Replace in-memory state with SQLite/PostgreSQL
   - Persist sessions across server restarts

7. **Multiple Model Support:**
   - VGGFace2, FaceNet for comparison
   - Ensemble voting for better accuracy

## 📜 License & Attribution

- **DeepFace:** Open-source library by Serengil
- **RetinaFace:** CVPR 2019 detector
- **ArcFace:** Additive Angular Margin loss for face recognition

---

**Last Updated:** May 8, 2024  
**Version:** 1.0 Complete  
**Status:** Production-Ready
