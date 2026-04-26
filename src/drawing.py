import cv2
import numpy as np

from src.tracking import Track


COLORS = {
    "confirmed": (0, 180, 0),
    "candidate": (0, 220, 255),
    "unknown": (0, 0, 255),
}


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> None:
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        color = COLORS.get(track.status, COLORS["unknown"])
        label = (
            f"#{track.track_id} {track.display_name} "
            f"{track.status} {track.avg_score:.2f} v:{track.vote_count}"
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
