from dataclasses import dataclass
from enum import IntEnum
import os
from pathlib import Path
import shutil
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen

import cv2
import mediapipe as mp


POSE_MODEL_ENV_VAR = "CRUXES_POSE_MODEL_PATH"
DEFAULT_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
DEFAULT_POSE_MODEL_PATH = (
    Path.home() / ".cache" / "cruxes" / "mediapipe" / "pose_landmarker_full.task"
)
VISIBILITY_THRESHOLD = 0.5
PRESENCE_THRESHOLD = 0.5


class PoseLandmark(IntEnum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
]


@dataclass
class NormalizedPoseLandmark:
    x: float
    y: float
    z: float
    visibility: Optional[float] = None
    presence: Optional[float] = None

    @classmethod
    def from_landmark(cls, landmark):
        return cls(
            x=landmark.x,
            y=landmark.y,
            z=landmark.z,
            visibility=getattr(landmark, "visibility", None),
            presence=getattr(landmark, "presence", None),
        )


@dataclass
class PoseResult:
    pose_landmarks: Optional[list[NormalizedPoseLandmark]]


class PoseDetector:
    def __init__(self):
        self._backend = None
        self._detector = None
        self._init_backend()

    def _init_backend(self):
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self._backend = "solutions"
            self._detector = mp.solutions.pose.Pose()
            return

        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        model_path = resolve_pose_model_path()
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._backend = "tasks"
        self._detector = vision.PoseLandmarker.create_from_options(options)

    def process(self, frame_bgr, timestamp_ms=None):
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._backend == "solutions":
            result = self._detector.process(image_rgb)
            if not result.pose_landmarks:
                return PoseResult(pose_landmarks=None)
            return PoseResult(
                pose_landmarks=[
                    NormalizedPoseLandmark.from_landmark(landmark)
                    for landmark in result.pose_landmarks.landmark
                ]
            )

        if timestamp_ms is None:
            raise ValueError("timestamp_ms is required for the MediaPipe Tasks backend")

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)
        if not result.pose_landmarks:
            return PoseResult(pose_landmarks=None)
        return PoseResult(
            pose_landmarks=[
                NormalizedPoseLandmark.from_landmark(landmark)
                for landmark in result.pose_landmarks[0]
            ]
        )

    def close(self):
        if self._detector is not None:
            self._detector.close()


def resolve_pose_model_path():
    env_model_path = os.environ.get(POSE_MODEL_ENV_VAR)
    if env_model_path:
        if os.path.isfile(env_model_path):
            return env_model_path
        raise FileNotFoundError(
            f"{POSE_MODEL_ENV_VAR} is set, but the file does not exist: {env_model_path}"
        )

    if DEFAULT_POSE_MODEL_PATH.is_file():
        return str(DEFAULT_POSE_MODEL_PATH)

    return download_default_pose_model(DEFAULT_POSE_MODEL_PATH)


def download_default_pose_model(destination_path):
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_suffix(destination_path.suffix + ".tmp")

    print(f"Downloading MediaPipe pose model to {destination_path}...")
    try:
        with (
            urlopen(DEFAULT_POSE_MODEL_URL) as response,
            open(temporary_path, "wb") as file_obj,
        ):
            shutil.copyfileobj(response, file_obj)
        os.replace(temporary_path, destination_path)
    except (OSError, URLError) as exc:
        if temporary_path.exists():
            temporary_path.unlink()
        raise RuntimeError(
            "MediaPipe on this Python build requires a Pose Landmarker model. "
            f"Failed to download the default model from {DEFAULT_POSE_MODEL_URL}. "
            f"Set {POSE_MODEL_ENV_VAR} to a local .task file or rerun with network access."
        ) from exc

    return str(destination_path)


def draw_pose_landmarks(image, landmarks, color=(255, 255, 255), thickness=2):
    if not landmarks:
        return

    image_height, image_width = image.shape[:2]
    coordinates = {}
    for idx, landmark in enumerate(landmarks):
        visibility = getattr(landmark, "visibility", None)
        presence = getattr(landmark, "presence", None)
        if visibility is not None and visibility < VISIBILITY_THRESHOLD:
            continue
        if presence is not None and presence < PRESENCE_THRESHOLD:
            continue
        if not (0.0 <= landmark.x <= 1.0 and 0.0 <= landmark.y <= 1.0):
            continue
        coordinates[idx] = (
            min(int(landmark.x * image_width), image_width - 1),
            min(int(landmark.y * image_height), image_height - 1),
        )

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx in coordinates and end_idx in coordinates:
            cv2.line(
                image, coordinates[start_idx], coordinates[end_idx], color, thickness
            )
