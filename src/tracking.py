import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.recognition import MatchResult


BBox = tuple[int, int, int, int]


@dataclass
class Track:
    track_id: int
    bbox: BBox
    history_size: int
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    predictions: deque[str] = field(init=False)
    scores: deque[float] = field(init=False)
    frames_observed: int = 0
    unknown_frames: int = 0
    status: str = "unknown"
    display_name: str = "Unknown"
    avg_score: float = 0.0
    vote_count: int = 0
    last_logged_status: str | None = None
    failure_logged: bool = False

    def __post_init__(self) -> None:
        self.predictions = deque(maxlen=self.history_size)
        self.scores = deque(maxlen=self.history_size)

    def update(self, bbox: BBox, match: MatchResult | None, config: dict[str, Any]) -> str:
        previous_status = self.status
        self.bbox = bbox
        self.last_seen = time.time()
        self.frames_observed += 1

        if match is None:
            self.unknown_frames += 1
        else:
            self.unknown_frames = 0
            self.predictions.append(match.name)
            self.scores.append(match.score)

        self._refresh_status(config)
        return previous_status

    def mark_missing(self) -> None:
        self.unknown_frames += 1

    def _refresh_status(self, config: dict[str, Any]) -> None:
        recognition_config = config["recognition"]
        min_votes = int(recognition_config["min_votes_to_confirm"])
        min_average_score = float(recognition_config["min_average_score"])
        max_unknown_frames = int(recognition_config["max_unknown_frames"])

        if not self.predictions:
            self.status = "unknown"
            self.display_name = "Unknown"
            self.avg_score = 0.0
            self.vote_count = 0
            return

        votes = Counter(self.predictions)
        best_name, vote_count = votes.most_common(1)[0]
        matching_scores = [
            score
            for name, score in zip(self.predictions, self.scores, strict=False)
            if name == best_name
        ]
        avg_score = float(np.mean(matching_scores)) if matching_scores else 0.0

        self.display_name = best_name
        self.avg_score = avg_score
        self.vote_count = vote_count

        if vote_count >= min_votes and avg_score >= min_average_score:
            self.status = "confirmed"
        elif self.unknown_frames >= max_unknown_frames:
            self.status = "unknown"
            self.display_name = "Unknown"
        else:
            self.status = "candidate"


class SimpleTracker:
    def __init__(self, config: dict[str, Any]) -> None:
        tracking_config = config["tracking"]
        self.max_distance = float(tracking_config.get("max_center_distance", 120))
        self.max_missing_frames = int(tracking_config.get("max_missing_frames", 30))
        self.history_size = int(config["recognition"]["recognition_window_frames"])
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(
        self,
        detections: list[tuple[BBox, MatchResult | None]],
        config: dict[str, Any],
    ) -> list[tuple[Track, str]]:
        events: list[tuple[Track, str]] = []
        unmatched_track_ids = set(self.tracks)

        for bbox, match in detections:
            track = self._match_track(bbox, unmatched_track_ids)

            if track is None:
                track = self._create_track(bbox)
            else:
                unmatched_track_ids.discard(track.track_id)

            previous_status = track.update(bbox, match, config)
            events.append((track, previous_status))

        for track_id in list(unmatched_track_ids):
            self.tracks[track_id].mark_missing()

        self._remove_stale_tracks()
        return events

    def active_tracks(self) -> list[Track]:
        return sorted(self.tracks.values(), key=lambda track: track.track_id)

    def _create_track(self, bbox: BBox) -> Track:
        track = Track(
            track_id=self._next_id,
            bbox=bbox,
            history_size=self.history_size,
        )
        self.tracks[track.track_id] = track
        self._next_id += 1
        return track

    def _match_track(self, bbox: BBox, candidate_ids: set[int]) -> Track | None:
        best_track: Track | None = None
        best_distance = self.max_distance

        for track_id in candidate_ids:
            track = self.tracks[track_id]
            distance = _center_distance(bbox, track.bbox)
            if distance < best_distance:
                best_distance = distance
                best_track = track

        return best_track

    def _remove_stale_tracks(self) -> None:
        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.unknown_frames > self.max_missing_frames
        ]

        for track_id in stale_ids:
            del self.tracks[track_id]


def face_bbox(face: Any) -> BBox:
    x1, y1, x2, y2 = map(int, face.bbox)
    return x1, y1, x2, y2


def _center_distance(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    center_a = ((ax1 + ax2) / 2, (ay1 + ay2) / 2)
    center_b = ((bx1 + bx2) / 2, (by1 + by2) / 2)
    return float(np.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1]))
