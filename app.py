"""
Interview.AI face verification backend.

FastAPI is the server implementation.
"""
import uvicorn

from app_fastapi import (
    app,
    ARCFACE_THRESHOLD,
    DETECTOR,
    MAX_FAIL_STREAK,
    MAX_WARNINGS,
    MIN_BRIGHTNESS,
    MAX_BRIGHTNESS,
    MODEL,
)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  INTERVIEW.AI - FACE VERIFICATION BACKEND (FASTAPI)")
    print("=" * 70)
    print(f"  Model:        {MODEL}")
    print(f"  Detector:     {DETECTOR}")
    print(f"  Threshold:    {ARCFACE_THRESHOLD}")
    print(f"  Brightness:   {MIN_BRIGHTNESS}-{MAX_BRIGHTNESS}")
    print(f"  Max Warnings: {MAX_WARNINGS}")
    print(f"  Max Fail Streak: {MAX_FAIL_STREAK}")
    print("\n  Endpoints:")
    print("    GET  /health              - Backend status")
    print("    POST /start_interview     - Register candidate & capture reference")
    print("    POST /verify_frame        - Verify webcam frame during interview")
    print("    POST /terminate_interview - Manually end interview")
    print("    GET  /session_logs/<id>   - Retrieve session audit log")
    print("    POST /reset_session       - Reset session state")
    print("\n  Serving: http://127.0.0.1:5000")
    print("\n" + "=" * 70 + "\n")

    uvicorn.run(app, host="127.0.0.1", port=5000)
