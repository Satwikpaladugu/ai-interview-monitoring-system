import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app_fastapi
from app_fastapi import app, candidate_store, session_state

client = TestClient(app)


def image_bytes(value=128, width=120, height=120):
    image = np.full((height, width, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def upload(name="frame.jpg", value=128):
    return {"file": (name, io.BytesIO(image_bytes(value)), "image/jpeg")}


def face_result(embedding=None, facial_area=None):
    return {
        "embedding": embedding or [1.0, 0.0, 0.0],
        "face_confidence": 0.99,
        "facial_area": facial_area
        or {
            "x": 30,
            "y": 30,
            "w": 80,
            "h": 80,
            "left_eye": (45, 55),
            "right_eye": (85, 55),
        },
    }


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "uploads").mkdir()
    (tmp_path / "logs").mkdir()
    candidate_store.clear()
    session_state.clear()
    yield
    candidate_store.clear()
    session_state.clear()


def register_candidate(candidate_id="candidate-1", embedding=None):
    candidate_store[candidate_id] = {
        "embedding": embedding or [1.0, 0.0, 0.0],
        "profile_image": f"reference_{candidate_id}.jpg",
        "candidate_name": "Test Candidate",
        "loaded_at": "2026-05-21 10:00:00",
        "face_size": "100x100px",
    }
    session_state[candidate_id] = {
        "warning_count": 0,
        "fail_streak": 0,
        "terminated": False,
        "check_count": 0,
    }


def test_health_success():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert response.json()["registered_candidates"] == 0


def test_start_interview_success(monkeypatch):
    monkeypatch.setattr(
        app_fastapi.DeepFace,
        "represent",
        lambda **kwargs: [face_result(facial_area={"x": 10, "y": 10, "w": 100, "h": 100})],
    )

    response = client.post(
        "/start_interview",
        data={"candidate_id": "candidate-1", "candidate_name": "Test Candidate"},
        files=upload("reference.jpg"),
    )

    data = response.json()
    assert response.status_code == 200
    assert data["success"] is True
    assert data["candidate_id"] == "candidate-1"
    assert candidate_store["candidate-1"]["candidate_name"] == "Test Candidate"
    assert session_state["candidate-1"]["check_count"] == 0


@pytest.mark.parametrize(
    ("data", "files", "expected_status", "expected_error"),
    [
        ({}, upload("reference.jpg"), 400, "Missing candidate_id or candidate_name"),
        (
            {"candidate_id": "candidate-1"},
            upload("reference.jpg"),
            400,
            "Missing candidate_id or candidate_name",
        ),
        (
            {"candidate_id": "candidate-1", "candidate_name": "Test Candidate"},
            None,
            400,
            "No reference image provided",
        ),
        (
            {"candidate_id": "candidate-1", "candidate_name": "Test Candidate"},
            {"file": ("", io.BytesIO(image_bytes()), "image/jpeg")},
            422,
            None,
        ),
    ],
)
def test_start_interview_missing_input_errors(data, files, expected_status, expected_error):
    response = client.post("/start_interview", data=data, files=files)

    assert response.status_code == expected_status
    if expected_error:
        assert response.json()["error"] == expected_error


@pytest.mark.parametrize(
    ("deepface_result", "expected_error"),
    [
        (ValueError("no face"), "No face found in reference image"),
        ([], "No face detected in reference image"),
        ([face_result(), face_result()], "Multiple faces in reference image"),
        ([face_result(facial_area={"x": 10, "y": 10, "w": 40, "h": 40})], "Face in reference image is too small"),
    ],
)
def test_start_interview_face_processing_errors(monkeypatch, deepface_result, expected_error):
    def fake_represent(**kwargs):
        if isinstance(deepface_result, Exception):
            raise deepface_result
        return deepface_result

    monkeypatch.setattr(app_fastapi.DeepFace, "represent", fake_represent)

    response = client.post(
        "/start_interview",
        data={"candidate_id": "candidate-1", "candidate_name": "Test Candidate"},
        files=upload("reference.jpg"),
    )

    assert response.status_code == 400
    assert response.json()["error"] == expected_error


def test_verify_frame_success_strong_match(monkeypatch):
    register_candidate()
    monkeypatch.setattr(app_fastapi, "check_face_pose", lambda img: (True, [face_result()]))

    response = client.post(
        "/verify_frame",
        data={"candidate_id": "candidate-1"},
        files=upload("frame.jpg", value=128),
    )

    data = response.json()
    assert response.status_code == 200
    assert data["verified"] is True
    assert data["status"] == "Strong Match"
    assert data["terminate"] is False
    assert data["check_count"] == 1


@pytest.mark.parametrize(
    ("data", "files", "expected_error"),
    [
        ({}, upload("frame.jpg"), "Missing candidate_id"),
        ({"candidate_id": "unknown"}, upload("frame.jpg"), "Candidate not registered"),
        ({"candidate_id": "candidate-1"}, None, "No frame provided"),
        (
            {"candidate_id": "candidate-1"},
            {"file": ("frame.jpg", io.BytesIO(b"not an image"), "image/jpeg")},
            "Failed to read frame",
        ),
    ],
)
def test_verify_frame_missing_input_errors(data, files, expected_error):
    register_candidate()

    response = client.post("/verify_frame", data=data, files=files)

    assert response.status_code == 400
    assert response.json()["error"] == expected_error


@pytest.mark.parametrize(
    ("brightness", "expected_status"),
    [
        (0, "too_dark"),
        (255, "too_bright"),
    ],
)
def test_verify_frame_brightness_edges(brightness, expected_status):
    register_candidate()

    response = client.post(
        "/verify_frame",
        data={"candidate_id": "candidate-1"},
        files=upload("frame.jpg", value=brightness),
    )

    data = response.json()
    assert response.status_code == 200
    assert data["status"] == expected_status
    assert data["verified"] is False
    assert data["warning_count"] == 1


@pytest.mark.parametrize(
    ("pose_result", "expected_status", "expected_suggestion"),
    [
        ((False, []), "no_face", "Ensure your face is visible to the camera"),
        ((True, [face_result(), face_result()]), "multiple_faces", "Only the candidate should be visible to the camera"),
        (
            (True, [face_result(facial_area={"x": 0, "y": 30, "w": 40, "h": 80})]),
            "face_not_centered",
            "Move closer to camera",
        ),
        (
            (
                True,
                [
                    face_result(
                        facial_area={
                            "x": 30,
                            "y": 30,
                            "w": 80,
                            "h": 80,
                            "left_eye": (50, 55),
                            "right_eye": (55, 55),
                        }
                    )
                ],
            ),
            "side_face",
            "Look directly at the camera",
        ),
    ],
)
def test_verify_frame_face_detection_edges(monkeypatch, pose_result, expected_status, expected_suggestion):
    register_candidate()
    monkeypatch.setattr(app_fastapi, "check_face_pose", lambda img: pose_result)

    response = client.post(
        "/verify_frame",
        data={"candidate_id": "candidate-1"},
        files=upload("frame.jpg", value=128),
    )

    data = response.json()
    assert response.status_code == 200
    assert data["status"] == expected_status
    assert data["verified"] is False
    assert data["warning_count"] == 1
    assert expected_suggestion in data["suggestions"]


def test_warning_termination_after_three_warnings():
    register_candidate()

    response = None
    for _ in range(3):
        response = client.post(
            "/verify_frame",
            data={"candidate_id": "candidate-1"},
            files=upload("frame.jpg", value=0),
        )

    data = response.json()
    assert data["terminate"] is True
    assert data["reason"] == "Too many warnings"
    assert data["warning_count"] == 3
    assert session_state["candidate-1"]["terminated"] is True


def test_identity_failure_termination_after_three_failures(monkeypatch):
    register_candidate(embedding=[1.0, 0.0, 0.0])
    monkeypatch.setattr(
        app_fastapi,
        "check_face_pose",
        lambda img: (True, [face_result(embedding=[0.0, 1.0, 0.0])]),
    )

    response = None
    for _ in range(3):
        response = client.post(
            "/verify_frame",
            data={"candidate_id": "candidate-1"},
            files=upload("frame.jpg", value=128),
        )

    data = response.json()
    assert data["terminate"] is True
    assert data["reason"] == "Identity verification failed repeatedly"
    assert data["fail_streak"] == 3
    assert session_state["candidate-1"]["terminated"] is True


def test_already_terminated_session_returns_terminated_status():
    register_candidate()
    session_state["candidate-1"]["terminated"] = True

    response = client.post(
        "/verify_frame",
        data={"candidate_id": "candidate-1"},
        files=upload("frame.jpg", value=128),
    )

    data = response.json()
    assert response.status_code == 200
    assert data["terminate"] is True
    assert data["status"] == "terminated"
    assert data["reason"] == "Session already terminated"


def test_terminate_interview_success():
    register_candidate()

    response = client.post(
        "/terminate_interview",
        data={"candidate_id": "candidate-1", "reason": "Manual test stop"},
    )

    data = response.json()
    assert response.status_code == 200
    assert data["success"] is True
    assert data["reason"] == "Manual test stop"
    assert session_state["candidate-1"]["terminated"] is True


@pytest.mark.parametrize(
    ("data", "expected_status", "expected_error"),
    [
        ({}, 400, "Missing candidate_id"),
        ({"candidate_id": "unknown"}, 404, "Session not found"),
    ],
)
def test_terminate_interview_errors(data, expected_status, expected_error):
    response = client.post("/terminate_interview", data=data)

    assert response.status_code == expected_status
    assert response.json()["error"] == expected_error


def test_session_logs_and_reset_success(monkeypatch):
    register_candidate()
    monkeypatch.setattr(app_fastapi, "check_face_pose", lambda img: (True, [face_result()]))

    verify_response = client.post(
        "/verify_frame",
        data={"candidate_id": "candidate-1"},
        files=upload("frame.jpg", value=128),
    )
    assert verify_response.status_code == 200

    logs_response = client.get("/session_logs/candidate-1")
    logs = logs_response.json()
    assert logs_response.status_code == 200
    assert logs["total_checks"] == 1
    assert logs["verified_checks"] == 1
    assert logs["integrity_score"] == 100.0
    assert logs["session_terminated"] is False
    assert len(logs["logs"]) == 1

    session_state["candidate-1"]["warning_count"] = 2
    session_state["candidate-1"]["fail_streak"] = 1

    reset_response = client.post("/reset_session", data={"candidate_id": "candidate-1"})
    reset = reset_response.json()
    assert reset_response.status_code == 200
    assert reset["success"] is True
    assert session_state["candidate-1"] == {
        "warning_count": 0,
        "fail_streak": 0,
        "terminated": False,
        "check_count": 0,
    }


def test_session_logs_empty_for_unknown_candidate():
    response = client.get("/session_logs/unknown")

    assert response.status_code == 200
    assert response.json() == {"candidate_id": "unknown", "logs": [], "total_checks": 0}


@pytest.mark.parametrize(
    ("data", "expected_status", "expected_error"),
    [
        ({}, 400, "Missing candidate_id"),
        ({"candidate_id": "unknown"}, 404, "Session not found"),
    ],
)
def test_reset_session_errors(data, expected_status, expected_error):
    response = client.post("/reset_session", data=data)

    assert response.status_code == expected_status
    assert response.json()["error"] == expected_error
