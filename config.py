"""
config.py — Edit this file to configure each game.
"""

import os
from dotenv import load_dotenv
import supervision as sv

load_dotenv()

# ── API Keys (set in .env) ────────────────────────────────────────
HF_TOKEN         = os.getenv("HF_TOKEN", "")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")

os.environ["HF_TOKEN"]        = HF_TOKEN
os.environ["ROBOFLOW_API_KEY"] = ROBOFLOW_API_KEY
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"

# ── Video ─────────────────────────────────────────────────────────
SOURCE_VIDEO_FILENAME = "dartpenn3.mp4"

# ── Teams ─────────────────────────────────────────────────────────
# After running team clustering, inspect the two grids and confirm
# which cluster ID (0 or 1) maps to which team.
TEAM_NAMES = {
    0: "Dartmouth Big Green",
    1: "Pennsylvania Quakers",
}

TEAM_COLORS = {
    "Pennsylvania Quakers": "#b22025",
    "Dartmouth Big Green":  "#005e2e",
}

# ── Rosters { "jersey_number": "LastName" } ───────────────────────
TEAM_ROSTERS = {
    "Dartmouth Big Green": {
        "33": "Munro",
        "22": "Williams",
        "21": "Mitchell-Day",
        "30": "Amundsen",
        "2":  "Thomas",
        "10": "Strelnikov",
        "11": "Hiatt",
        "5":  "McNamee",
        "23": "Abusara",
        "0":  "Brown",
    },
    "Pennsylvania Quakers": {
        "25": "Gerhart",
        "7":  "Zanoni",
        "23": "Roberts",
        "12": "Power",
        "0":  "Levine",
        "30": "Lueth",
        "35": "Polonowski",
        "13": "Scantlebury",
        "33": "Oberti",
        "4":  "Jones",
        "5":  "Thrower",
    },
}

# ── Model IDs (Roboflow) ──────────────────────────────────────────
PLAYER_DETECTION_MODEL_ID         = "basketball-player-detection-3-ycjdo/4"
PLAYER_DETECTION_MODEL_CONFIDENCE = 0.4
PLAYER_DETECTION_MODEL_IOU        = 0.9

NUMBER_RECOGNITION_MODEL_ID     = "basketball-jersey-numbers-ocr/3"
NUMBER_RECOGNITION_MODEL_PROMPT = "Read the number."

KEYPOINT_DETECTION_MODEL_ID                = "basketball-court-detection-2/14"
KEYPOINT_DETECTION_MODEL_CONFIDENCE        = 0.3
KEYPOINT_DETECTION_MODEL_ANCHOR_CONFIDENCE = 0.5

# ── SAM2 ──────────────────────────────────────────────────────────
SAM2_CHECKPOINT = "segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_l.yaml"

# ── Class IDs ─────────────────────────────────────────────────────
PLAYER_CLASS_IDS        = [3, 4, 5, 6, 7]
NUMBER_CLASS_ID         = 2
BALL_IN_BASKET_CLASS_ID = 1
JUMP_SHOT_CLASS_ID      = 5
LAYUP_DUNK_CLASS_ID     = 6

# ── Annotation palette ────────────────────────────────────────────
COLOR = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2",
    "#ff8080", "#b266ff", "#9999ff", "#66ffff", "#33ff99",
    "#66ff66", "#99ff00",
])
