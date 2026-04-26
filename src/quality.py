from typing import Any

import cv2
import numpy as np

from src.tracking import BBox


def is_face_usable(frame_bgr: np.ndarray, bbox: BBox, face: Any, config: dict[str, Any]) -> bool:
    quality_config = config.get("quality", {})
    min_width = int(quality_config.get("min_face_width", 80))
    min_height = int(quality_config.get("min_face_height", 80))
    min_blur_score = float(quality_config.get("min_blur_score", 40.0))
    min_detection_score = quality_config.get("min_detection_score")

    x1, y1, x2, y2 = _clip_bbox(bbox, frame_bgr.shape)
    width = x2 - x1
    height = y2 - y1

    if width < min_width or height < min_height:
        return False

    if min_detection_score is not None:
        det_score = getattr(face, "det_score", None)
        if det_score is not None and float(det_score) < float(min_detection_score):
            return False

    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return blur_score >= min_blur_score


def _clip_bbox(bbox: BBox, shape: tuple[int, ...]) -> BBox:
    height, width = shape[:2]
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )
