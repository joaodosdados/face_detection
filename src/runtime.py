import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from src.config import ensure_runtime_dirs, setup_cuda_paths
from src.drawing import draw_fps, draw_tracks
from src.logging_utils import RecognitionLogger, read_recent_events
from src.model import detect_faces, load_model
from src.quality import is_face_usable
from src.recognition import MatchResult, best_match
from src.references import ReferenceIdentity, load_reference_faces
from src.tracking import SimpleTracker, Track, face_bbox
from src.video import open_video_source


@dataclass
class RuntimeMetrics:
    frame_number: int = 0
    average_fps: float = 0.0
    active_tracks: int = 0
    confirmed_recognitions: int = 0
    registered_people: int = 0
    video_source: str = ""
    runtime_mode: str = ""
    running: bool = False


@dataclass
class RuntimeSnapshot:
    metrics: RuntimeMetrics
    tracks: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, str]] = field(default_factory=list)


class RecognitionRuntime:
    def __init__(self, config: dict[str, Any], display_window: bool = False) -> None:
        self.config = config
        self.display_window = display_window
        self.metrics = RuntimeMetrics()
        self.latest_frame: np.ndarray | None = None
        self.latest_tracks: list[Track] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self.event_logger: RecognitionLogger | None = None

    def start_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._capture is not None:
            self._capture.release()
        with self._lock:
            self.metrics.running = False
        if (
            self._thread
            and self._thread.is_alive()
            and threading.current_thread() is not self._thread
        ):
            self._thread.join(timeout=2)

    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        setup_cuda_paths(self.config)
        ensure_runtime_dirs(self.config)

        app, runtime_mode = load_model(self.config)
        references = load_reference_faces(app, self.config)
        capture, video_label = open_video_source(self.config)
        self._capture = capture

        tracker = SimpleTracker(self.config)
        self.event_logger = RecognitionLogger(self.config)

        recognition_config = self.config["recognition"]
        ui_config = self.config["ui"]
        process_every_n_frames = max(1, int(recognition_config["process_every_n_frames"]))
        recognition_window_frames = int(recognition_config["recognition_window_frames"])
        window_name = ui_config.get("window_name", "Biometric Access Control MVP")
        show_fps = bool(ui_config.get("show_fps", True))

        fps_samples: deque[float] = deque(maxlen=60)
        previous_time = time.time()

        with self._lock:
            self.metrics = RuntimeMetrics(
                registered_people=len(references),
                video_source=video_label,
                runtime_mode=runtime_mode,
                running=True,
            )

        try:
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    print("[WARN] Could not read frame from video source.")
                    break

                with self._lock:
                    self.metrics.frame_number += 1
                    frame_number = self.metrics.frame_number

                if frame_number % process_every_n_frames == 0:
                    faces = detect_faces(app, frame)
                    detections = build_detections(frame, faces, references, self.config)
                    status_events = tracker.update(detections, self.config)

                    for track, previous_status in status_events:
                        became_confirmed = self.event_logger.maybe_log_status_change(
                            track=track,
                            previous_status=previous_status,
                            frame_number=frame_number,
                            frame_bgr=frame,
                        )
                        if became_confirmed:
                            with self._lock:
                                self.metrics.confirmed_recognitions += 1

                        self.event_logger.maybe_log_failure(
                            track=track,
                            frame_number=frame_number,
                            window_frames=recognition_window_frames,
                            frame_bgr=frame,
                        )

                current_time = time.time()
                elapsed = current_time - previous_time
                previous_time = current_time

                if elapsed > 0:
                    fps_samples.append(1 / elapsed)

                average_fps = float(np.mean(fps_samples)) if fps_samples else 0.0
                active_tracks = tracker.active_tracks()

                draw_tracks(frame, active_tracks)
                if show_fps:
                    draw_fps(frame, average_fps)

                with self._lock:
                    self.metrics.average_fps = average_fps
                    self.metrics.active_tracks = len(active_tracks)
                    self.latest_tracks = active_tracks
                    self.latest_frame = frame.copy()

                if self.display_window:
                    _maybe_print_runtime_metrics(
                        frame_number=frame_number,
                        active_tracks=len(active_tracks),
                        average_fps=average_fps,
                        confirmed_recognitions=self.metrics.confirmed_recognitions,
                    )

                if self.display_window:
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            capture.release()
            self._capture = None
            if self.display_window:
                cv2.destroyAllWindows()
            with self._lock:
                self.metrics.running = False

    def get_frame_jpeg(self) -> bytes | None:
        with self._lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()

        if frame is None:
            return None

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return None
        return buffer.tobytes()

    def snapshot(self) -> RuntimeSnapshot:
        with self._lock:
            metrics = RuntimeMetrics(**self.metrics.__dict__)
            tracks = [_track_to_dict(track) for track in self.latest_tracks]

        events = read_recent_events(self.config, limit=25)
        return RuntimeSnapshot(metrics=metrics, tracks=tracks, events=events)


def build_detections(
    frame_bgr: np.ndarray,
    faces: list[Any],
    references: list[ReferenceIdentity],
    config: dict[str, Any],
) -> list[tuple[tuple[int, int, int, int], MatchResult | None]]:
    recognition_config = config["recognition"]
    min_candidate_score = float(
        recognition_config.get(
            "candidate_min_score",
            max(0.0, float(recognition_config["min_average_score"]) - 0.10),
        )
    )

    detections: list[tuple[tuple[int, int, int, int], MatchResult | None]] = []

    for face in faces:
        bbox = face_bbox(face)

        if not is_face_usable(frame_bgr, bbox, face, config):
            continue

        match = best_match(face.embedding, references, min_candidate_score)
        detections.append((bbox, match))

    return detections


def _track_to_dict(track: Track) -> dict[str, Any]:
    x1, y1, x2, y2 = track.bbox
    return {
        "track_id": track.track_id,
        "name": track.display_name,
        "status": track.status,
        "avg_score": round(track.avg_score, 4),
        "votes": track.vote_count,
        "frames_observed": track.frames_observed,
        "unknown_frames": track.unknown_frames,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }


def _maybe_print_runtime_metrics(
    frame_number: int,
    active_tracks: int,
    average_fps: float,
    confirmed_recognitions: int,
    interval_frames: int = 90,
) -> None:
    if frame_number % interval_frames != 0:
        return

    print(
        "metrics | "
        f"frame={frame_number} | "
        f"active_tracks={active_tracks} | "
        f"avg_fps={average_fps:.1f} | "
        f"confirmed={confirmed_recognitions}"
    )
