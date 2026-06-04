"""
Basketball Player Tracker
=========================
Tracks players across a basketball game video and produces:
  - Per-player masks with team colors and name labels
  - A top-down court minimap showing real-time player positions
  - Shot-event detection (jump shots, layups, dunks, made/miss)

Usage:
    python main.py --video path/to/game.mp4 [--mode all|track|map|shots]
"""

from __future__ import annotations
import argparse
from pathlib import Path

from config import (
    SOURCE_VIDEO_FILENAME, TEAM_NAMES, TEAM_COLORS, TEAM_ROSTERS,
    PLAYER_DETECTION_MODEL_CONFIDENCE, PLAYER_DETECTION_MODEL_IOU,
    PLAYER_CLASS_IDS, NUMBER_CLASS_ID, BALL_IN_BASKET_CLASS_ID,
    JUMP_SHOT_CLASS_ID, LAYUP_DUNK_CLASS_ID, COLOR,
    KEYPOINT_DETECTION_MODEL_CONFIDENCE, KEYPOINT_DETECTION_MODEL_ANCHOR_CONFIDENCE,
    SAM2_CHECKPOINT, SAM2_CONFIG,
)
from tracker import run_player_recognition, run_court_minimap, run_shot_detection
from models import load_models


def parse_args():
    parser = argparse.ArgumentParser(description="Basketball Player Tracker")
    parser.add_argument("--video", type=str, default=SOURCE_VIDEO_FILENAME,
                        help="Path to source video file")
    parser.add_argument("--mode", choices=["all", "track", "map", "shots"],
                        default="all", help="Which pipeline(s) to run")
    parser.add_argument("--font", type=str, default="fonts/Staatliches-Regular.ttf",
                        help="Path to font file for labels")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print("Loading models...")
    models = load_models()

    if args.mode in ("all", "track"):
        print("\n=== Running Player Recognition ===")
        run_player_recognition(video_path, models, args.font)

    if args.mode in ("all", "map"):
        print("\n=== Running Court Minimap ===")
        run_court_minimap(video_path, models)

    if args.mode in ("all", "shots"):
        print("\n=== Running Shot Detection ===")
        run_shot_detection(video_path, models)

    print("\nDone.")


if __name__ == "__main__":
    main()
