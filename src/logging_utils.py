import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.tracking import Track


FIELDNAMES = [
    "timestamp",
    "track_id",
    "name",
    "status",
    "avg_score",
    "votes",
    "frame_number",
    "x1",
    "y1",
    "x2",
    "y2",
    "snapshot_path",
]


class RecognitionLogger:
    def __init__(self, config: dict[str, Any]) -> None:
        logging_config = config.get("logging", {})
        directory = Path(logging_config.get("directory", "logs"))
        filename = logging_config.get("events_file", "recognition_events.csv")
        self.path = directory / filename
        self.snapshots_dir = directory / "snapshots"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def maybe_log_status_change(
        self,
        track: Track,
        previous_status: str,
        frame_number: int,
        frame_bgr: np.ndarray | None = None,
    ) -> bool:
        if track.status == "unknown" and previous_status == "unknown":
            return False

        if track.status == previous_status and track.last_logged_status == track.status:
            return False

        if track.status == "candidate" and previous_status == "candidate":
            return False

        self.log(track, frame_number, frame_bgr)
        track.last_logged_status = track.status
        return track.status == "confirmed"

    def maybe_log_failure(
        self,
        track: Track,
        frame_number: int,
        window_frames: int,
        frame_bgr: np.ndarray | None = None,
    ) -> None:
        if track.failure_logged:
            return

        if track.status == "unknown" and track.frames_observed >= window_frames:
            self.log(track, frame_number, frame_bgr)
            track.failure_logged = True
            track.last_logged_status = track.status

    def log(
        self,
        track: Track,
        frame_number: int,
        frame_bgr: np.ndarray | None = None,
    ) -> None:
        x1, y1, x2, y2 = track.bbox
        snapshot_path = self._save_snapshot(track, frame_number, frame_bgr)
        row = {
            "timestamp": _now_iso(),
            "track_id": track.track_id,
            "name": track.display_name,
            "status": track.status,
            "avg_score": f"{track.avg_score:.4f}",
            "votes": track.vote_count,
            "frame_number": frame_number,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "snapshot_path": snapshot_path,
        }

        with self.path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            writer.writerow(row)

    def _ensure_header(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                if reader.fieldnames == FIELDNAMES:
                    return

            with self.path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
                writer.writeheader()
                for row in rows:
                    writer.writerow({field: row.get(field, "") for field in FIELDNAMES})
            return

        with self.path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            writer.writeheader()

    def _save_snapshot(
        self,
        track: Track,
        frame_number: int,
        frame_bgr: np.ndarray | None,
    ) -> str:
        if frame_bgr is None:
            return ""

        x1, y1, x2, y2 = _clip_bbox(track.bbox, frame_bgr.shape)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return ""

        status_dir = self.snapshots_dir / track.status
        status_dir.mkdir(parents=True, exist_ok=True)

        filename = f"track_{track.track_id}_frame_{frame_number}.jpg"
        snapshot_path = status_dir / filename
        cv2.imwrite(str(snapshot_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        return snapshot_path.as_posix()


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


def read_recent_events(config: dict[str, Any], limit: int = 25) -> list[dict[str, str]]:
    logging_config = config.get("logging", {})
    directory = Path(logging_config.get("directory", "logs"))
    filename = logging_config.get("events_file", "recognition_events.csv")
    path = directory / filename

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    recent_rows = rows[-limit:][::-1]
    for row in recent_rows:
        row["snapshot_url"] = _snapshot_url(row.get("snapshot_path", ""))

    return recent_rows


def _clip_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def _snapshot_url(snapshot_path: str) -> str:
    normalized = snapshot_path.replace("\\", "/")
    prefix = "logs/snapshots/"

    if not normalized.startswith(prefix):
        return ""

    return "/snapshots/" + normalized.removeprefix(prefix)
