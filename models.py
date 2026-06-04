"""
models.py — Loads all inference models used by the tracker.
"""

from __future__ import annotations
import torch
from inference import get_model
from sam2.build_sam import build_sam2_camera_predictor

from config import (
    PLAYER_DETECTION_MODEL_ID,
    NUMBER_RECOGNITION_MODEL_ID,
    KEYPOINT_DETECTION_MODEL_ID,
    SAM2_CHECKPOINT,
    SAM2_CONFIG,
)


def load_models() -> dict:
    """Load and return all models as a dictionary."""
    print("  Loading player detection model...")
    player_model = get_model(model_id=PLAYER_DETECTION_MODEL_ID)

    print("  Loading jersey number OCR model...")
    number_model = get_model(model_id=NUMBER_RECOGNITION_MODEL_ID)

    print("  Loading court keypoint model...")
    keypoint_model = get_model(model_id=KEYPOINT_DETECTION_MODEL_ID)

    print("  Loading SAM2...")
    predictor = build_sam2_camera_predictor(SAM2_CONFIG, SAM2_CHECKPOINT)

    print("  All models loaded.")
    return {
        "player":   player_model,
        "number":   number_model,
        "keypoint": keypoint_model,
        "sam2":     predictor,
    }
