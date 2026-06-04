"""
tracker.py — SAM2 tracker wrapper and the three processing pipelines:
  1. run_player_recognition  — masked video with team colors + name labels
  2. run_court_minimap       — top-down court with player positions
  3. run_shot_detection      — frame-by-frame shot event classification
"""

from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

import supervision as sv
from sports import (
    clean_paths,
    ConsecutiveValueTracker,
    TeamClassifier,
    MeasurementUnit,
    ViewTransformer,
)
from sports.basketball import (
    CourtConfiguration,
    League,
    draw_court,
    draw_points_on_court,
    ShotEventTracker,
)

from config import (
    TEAM_NAMES, TEAM_COLORS, TEAM_ROSTERS,
    PLAYER_DETECTION_MODEL_CONFIDENCE, PLAYER_DETECTION_MODEL_IOU,
    PLAYER_CLASS_IDS, NUMBER_CLASS_ID, BALL_IN_BASKET_CLASS_ID,
    JUMP_SHOT_CLASS_ID, LAYUP_DUNK_CLASS_ID, COLOR,
    KEYPOINT_DETECTION_MODEL_CONFIDENCE, KEYPOINT_DETECTION_MODEL_ANCHOR_CONFIDENCE,
    NUMBER_RECOGNITION_MODEL_PROMPT,
)


# ─────────────────────────────────────────────────────────────────
# SAM2 Tracker
# ─────────────────────────────────────────────────────────────────

class SAM2Tracker:
    """Thin wrapper around the SAM2 camera predictor for player tracking."""

    def __init__(self, predictor) -> None:
        self.predictor = predictor
        self._prompted = False

    def prompt_first_frame(self, frame: np.ndarray, detections: sv.Detections) -> None:
        if len(detections) == 0:
            raise ValueError("detections must contain at least one box")
        if detections.tracker_id is None:
            detections.tracker_id = list(range(1, len(detections) + 1))

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.load_first_frame(frame)
            for xyxy, obj_id in zip(detections.xyxy, detections.tracker_id):
                self.predictor.add_new_prompt(
                    frame_idx=0,
                    obj_id=int(obj_id),
                    bbox=np.asarray([xyxy], dtype=np.float32),
                )
        self._prompted = True

    def propagate(self, frame: np.ndarray) -> sv.Detections:
        if not self._prompted:
            raise RuntimeError("Call prompt_first_frame before propagate")

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            tracker_ids, mask_logits = self.predictor.track(frame)

        tracker_ids = np.asarray(tracker_ids, dtype=np.int32)
        masks = (mask_logits > 0.0).cpu().numpy()
        masks = np.squeeze(masks).astype(bool)
        if masks.ndim == 2:
            masks = masks[None, ...]

        masks = np.array([
            sv.filter_segments_by_distance(mask, relative_distance=0.03, mode="edge")
            for mask in masks
        ])
        return sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks,
            tracker_id=tracker_ids,
        )

    def reset(self) -> None:
        self._prompted = False


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _compressed_path(path: Path) -> Path:
    return path.parent / f"{path.stem}-compressed{path.suffix}"


def _ffmpeg_compress(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-vcodec", "libx264", "-crf", "28", str(dst)],
        check=True,
    )


def coords_above_threshold(
    matrix: np.ndarray, threshold: float, sort_desc: bool = True
) -> List[Tuple[int, int]]:
    rows, cols = np.where(np.asarray(matrix) > threshold)
    pairs = list(zip(rows.tolist(), cols.tolist()))
    if sort_desc:
        pairs.sort(key=lambda rc: matrix[rc[0], rc[1]], reverse=True)
    return pairs


def build_label(number: str | None, team: int) -> str:
    if not number:
        return ""
    name = TEAM_ROSTERS[TEAM_NAMES[team]].get(number)
    return f"#{number} {name}" if name else f"#{number}"


def _cluster_teams(models: dict, source_dir: Path, stride: int = 30) -> TeamClassifier:
    crops = []
    for video_path in sv.list_files_with_extensions(source_dir, extensions=["mp4", "avi", "mov"]):
        for frame in tqdm(sv.get_video_frames_generator(source_path=str(video_path), stride=stride),
                          desc="Clustering teams"):
            result = models["player"].infer(
                frame,
                confidence=PLAYER_DETECTION_MODEL_CONFIDENCE,
                iou_threshold=PLAYER_DETECTION_MODEL_IOU,
                class_agnostic_nms=True,
            )[0]
            detections = sv.Detections.from_inference(result)
            detections = detections[np.isin(detections.class_id, PLAYER_CLASS_IDS)]
            for box in sv.scale_boxes(xyxy=detections.xyxy, factor=0.4):
                crops.append(sv.crop_image(frame, box))

    team_classifier = TeamClassifier(device="cuda")
    team_classifier.fit(crops)
    print(f"Clustered {len(crops)} crops into 2 teams.")
    return team_classifier


# ─────────────────────────────────────────────────────────────────
# Pipeline 1 — Player Recognition
# ─────────────────────────────────────────────────────────────────

def run_player_recognition(video_path: Path, models: dict, font_path: str) -> None:
    source_dir = video_path.parent
    team_classifier = _cluster_teams(models, source_dir)

    number_validator = ConsecutiveValueTracker(n_consecutive=3)
    team_validator   = ConsecutiveValueTracker(n_consecutive=1)

    frame_generator = sv.get_video_frames_generator(str(video_path))
    first_frame = next(frame_generator)

    result = models["player"].infer(
        first_frame,
        confidence=PLAYER_DETECTION_MODEL_CONFIDENCE,
        iou_threshold=PLAYER_DETECTION_MODEL_IOU,
    )[0]
    detections = sv.Detections.from_inference(result)
    detections = detections[np.isin(detections.class_id, PLAYER_CLASS_IDS)]
    detections.tracker_id = np.arange(1, len(detections.class_id) + 1)

    boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=0.4)
    first_crops = [sv.crop_image(first_frame, b) for b in boxes]
    TEAMS = np.array(team_classifier.predict(first_crops))
    team_validator.update(tracker_ids=detections.tracker_id, values=TEAMS)

    tracker = SAM2Tracker(models["sam2"])
    tracker.prompt_first_frame(first_frame, detections)

    frames_history, detections_history = [], []

    for index, frame in tqdm(enumerate(sv.get_video_frames_generator(str(video_path))),
                             desc="Tracking players"):
        player_detections = tracker.propagate(frame)
        frames_history.append(frame)
        detections_history.append(player_detections)

        if index % 5 == 0:
            frame_h, frame_w = frame.shape[:2]
            result = models["player"].infer(
                frame,
                confidence=PLAYER_DETECTION_MODEL_CONFIDENCE,
                iou_threshold=PLAYER_DETECTION_MODEL_IOU,
            )[0]
            number_detections = sv.Detections.from_inference(result)
            number_detections = number_detections[number_detections.class_id == NUMBER_CLASS_ID]
            number_detections.mask = sv.xyxy_to_mask(
                boxes=number_detections.xyxy, resolution_wh=(frame_w, frame_h)
            )

            iou = sv.mask_iou_batch(
                masks_true=player_detections.mask,
                masks_detection=number_detections.mask,
                overlap_metric=sv.OverlapMetric.IOS,
            )
            pairs = coords_above_threshold(iou, 0.9)
            if pairs:
                player_idx, number_idx = zip(*pairs)
                padded = sv.clip_boxes(
                    sv.pad_boxes(xyxy=number_detections.xyxy[list(number_idx)], px=10, py=10),
                    (frame_w, frame_h),
                )
                ocr_crops = [
                    sv.resize_image(sv.crop_image(frame, box), resolution_wh=(224, 224))
                    for box in padded
                ]
                numbers = [
                    models["number"].predict(crop, NUMBER_RECOGNITION_MODEL_PROMPT)[0]
                    for crop in ocr_crops
                ]
                matched_tracker_ids = player_detections.tracker_id[list(player_idx)]
                number_validator.update(tracker_ids=matched_tracker_ids, values=numbers)

    print(f"Processed {len(frames_history)} frames.")

    target_path = video_path.parent / f"{video_path.stem}-result{video_path.suffix}"
    compressed_path = _compressed_path(target_path)

    video_info  = sv.VideoInfo.from_video_path(str(video_path))
    team_colors = sv.ColorPalette.from_hex([TEAM_COLORS[TEAM_NAMES[0]], TEAM_COLORS[TEAM_NAMES[1]]])

    team_mask_annotator  = sv.MaskAnnotator(color=team_colors, opacity=0.5,
                                            color_lookup=sv.ColorLookup.INDEX)
    team_label_annotator = sv.RichLabelAnnotator(
        font_path=font_path,
        font_size=40,
        color=team_colors,
        text_color=sv.Color.WHITE,
        text_position=sv.Position.BOTTOM_CENTER,
        text_offset=(0, 10),
        color_lookup=sv.ColorLookup.INDEX,
    )

    with sv.VideoSink(str(target_path), video_info) as sink:
        for frame, det in tqdm(zip(frames_history, detections_history), desc="Writing video"):
            det = det[det.area > 100]
            teams   = np.array(team_validator.get_validated(tracker_ids=det.tracker_id)).astype(int)
            numbers = np.array(number_validator.get_validated(tracker_ids=det.tracker_id))
            labels  = [build_label(num, team) for num, team in zip(numbers, teams)]

            annotated = frame.copy()
            annotated = team_mask_annotator.annotate(scene=annotated, detections=det,
                                                     custom_color_lookup=teams)
            annotated = team_label_annotator.annotate(scene=annotated, detections=det,
                                                      labels=labels, custom_color_lookup=teams)
            sink.write_frame(annotated)

    _ffmpeg_compress(target_path, compressed_path)
    print(f"Saved: {compressed_path}")


# ─────────────────────────────────────────────────────────────────
# Pipeline 2 — Court Minimap
# ─────────────────────────────────────────────────────────────────

def run_court_minimap(video_path: Path, models: dict) -> None:
    source_dir = video_path.parent
    team_classifier = _cluster_teams(models, source_dir)

    config   = CourtConfiguration(league=League.NBA, measurement_unit=MeasurementUnit.FEET)
    video_xy = []

    frame_generator = sv.get_video_frames_generator(str(video_path))
    first_frame = next(frame_generator)

    result = models["player"].infer(
        first_frame,
        confidence=PLAYER_DETECTION_MODEL_CONFIDENCE,
        iou_threshold=PLAYER_DETECTION_MODEL_IOU,
    )[0]
    detections = sv.Detections.from_inference(result)
    detections = detections[np.isin(detections.class_id, PLAYER_CLASS_IDS)]
    detections.tracker_id = np.arange(1, len(detections.class_id) + 1)

    boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=0.4)
    TEAMS = np.array(team_classifier.predict([sv.crop_image(first_frame, b) for b in boxes]))

    tracker = SAM2Tracker(models["sam2"])
    tracker.prompt_first_frame(first_frame, detections)

    for frame in tqdm(sv.get_video_frames_generator(str(video_path)), desc="Mapping court"):
        player_dets = tracker.propagate(frame)

        result = models["keypoint"].infer(
            frame, confidence=KEYPOINT_DETECTION_MODEL_CONFIDENCE
        )[0]
        key_points     = sv.KeyPoints.from_inference(result)
        landmarks_mask = key_points.confidence[0] > KEYPOINT_DETECTION_MODEL_ANCHOR_CONFIDENCE

        if np.count_nonzero(landmarks_mask) >= 4:
            transformer = ViewTransformer(
                source=key_points[:, landmarks_mask].xy[0],
                target=np.array(config.vertices)[landmarks_mask],
            )
            frame_xy = player_dets.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            video_xy.append(transformer.transform_points(points=frame_xy))

    video_xy = np.array(video_xy)
    print(f"Collected court coordinates for {len(video_xy)} frames.")

    cleaned_xy, _ = clean_paths(
        video_xy,
        jump_sigma=3.5,
        min_jump_dist=0.6,
        max_jump_run=18,
        pad_around_runs=2,
        smooth_window=9,
        smooth_poly=2,
    )

    target_path    = video_path.parent / f"{video_path.stem}-map{video_path.suffix}"
    compressed_path = _compressed_path(target_path)

    court       = draw_court(config=config)
    court_h, court_w, _ = court.shape
    video_info  = sv.VideoInfo.from_video_path(str(video_path))
    video_info.width, video_info.height = court_w, court_h

    with sv.VideoSink(str(target_path), video_info) as sink:
        for frame_xy in tqdm(cleaned_xy, desc="Writing minimap"):
            court = draw_court(config=config)
            court = draw_points_on_court(
                config=config,
                xy=frame_xy[TEAMS == 0],
                fill_color=sv.Color.from_hex(TEAM_COLORS[TEAM_NAMES[0]]),
                court=court,
            )
            court = draw_points_on_court(
                config=config,
                xy=frame_xy[TEAMS == 1],
                fill_color=sv.Color.from_hex(TEAM_COLORS[TEAM_NAMES[1]]),
                court=court,
            )
            sink.write_frame(court)

    _ffmpeg_compress(target_path, compressed_path)
    print(f"Saved: {compressed_path}")


# ─────────────────────────────────────────────────────────────────
# Pipeline 3 — Shot Detection
# ─────────────────────────────────────────────────────────────────

def run_shot_detection(video_path: Path, models: dict) -> None:
    video_info = sv.VideoInfo.from_video_path(str(video_path))
    shot_event_tracker = ShotEventTracker(
        reset_time_frames=int(video_info.fps * 1.7),
        minimum_frames_between_starts=int(video_info.fps * 0.5),
        cooldown_frames_after_made=int(video_info.fps * 0.5),
    )

    box_annotator   = sv.BoxAnnotator(color=COLOR, thickness=2)
    label_annotator = sv.LabelAnnotator(color=COLOR, text_color=sv.Color.BLACK)

    for frame_index, frame in enumerate(
        tqdm(sv.get_video_frames_generator(str(video_path)), desc="Detecting shots")
    ):
        result = models["player"].infer(
            frame,
            confidence=PLAYER_DETECTION_MODEL_CONFIDENCE,
            iou_threshold=PLAYER_DETECTION_MODEL_IOU,
        )[0]
        detections = sv.Detections.from_inference(result)

        events = shot_event_tracker.update(
            frame_index=frame_index,
            has_jump_shot=len(detections[detections.class_id == JUMP_SHOT_CLASS_ID]) > 0,
            has_layup_dunk=len(detections[detections.class_id == LAYUP_DUNK_CLASS_ID]) > 0,
            has_ball_in_basket=len(detections[detections.class_id == BALL_IN_BASKET_CLASS_ID]) > 0,
        )

        if events:
            print(f"Frame {frame_index}: {events}")
