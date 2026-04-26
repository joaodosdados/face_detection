import csv
from pathlib import Path
from typing import Any

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
]


class RecognitionLogger:
    def __init__(self, config: dict[str, Any]) -> None:
        logging_config = config.get("logging", {})
        directory = Path(logging_config.get("directory", "logs"))
        filename = logging_config.get("events_file", "recognition_events.csv")
        self.path = directory / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def maybe_log_status_change(
        self,
        track: Track,
        previous_status: str,
        frame_number: int,
    ) -> bool:
        if track.status == "unknown" and previous_status == "unknown":
            return False

        if track.status == previous_status and track.last_logged_status == track.status:
            return False

        if track.status == "candidate" and previous_status == "candidate":
            return False

        self.log(track, frame_number)
        track.last_logged_status = track.status
        return track.status == "confirmed"

    def maybe_log_failure(self, track: Track, frame_number: int, window_frames: int) -> None:
        if track.failure_logged:
            return

        if track.status == "unknown" and track.frames_observed >= window_frames:
            self.log(track, frame_number)
            track.failure_logged = True
            track.last_logged_status = track.status

    def log(self, track: Track, frame_number: int) -> None:
        x1, y1, x2, y2 = track.bbox
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
        }

        with self.path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            writer.writerow(row)

    def _ensure_header(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            return

        with self.path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            writer.writeheader()


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().isoformat(timespec="seconds")
