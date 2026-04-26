import time
from collections import deque
from typing import Any

import cv2
import numpy as np

from src.config import ensure_runtime_dirs, load_config, setup_cuda_paths
from src.drawing import draw_fps, draw_tracks
from src.logging_utils import RecognitionLogger
from src.model import detect_faces, load_model
from src.quality import is_face_usable
from src.recognition import MatchResult, best_match
from src.references import load_reference_faces
from src.tracking import SimpleTracker, face_bbox
from src.video import open_video_source


def build_detections(
    frame_bgr: np.ndarray,
    faces: list[Any],
    references: list[Any],
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


def print_startup_metrics(
    references: list[Any],
    video_label: str,
    runtime_mode: str,
    config: dict[str, Any],
) -> None:
    print("\nRuntime metrics")
    print(f"- registered people: {len(references)}")
    print(f"- video source: {video_label}")
    print(f"- model runtime: {runtime_mode}")
    print(f"- recognition window: {config['recognition']['recognition_window_frames']} frames")
    print("- biometric access control: disabled; this MVP only identifies and logs events")


def maybe_print_runtime_metrics(
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


def run(config: dict[str, Any]) -> None:
    setup_cuda_paths(config)
    ensure_runtime_dirs(config)

    app, runtime_mode = load_model(config)
    references = load_reference_faces(app, config)
    capture, video_label = open_video_source(config)

    tracker = SimpleTracker(config)
    event_logger = RecognitionLogger(config)

    recognition_config = config["recognition"]
    ui_config = config["ui"]
    process_every_n_frames = max(1, int(recognition_config["process_every_n_frames"]))
    recognition_window_frames = int(recognition_config["recognition_window_frames"])
    window_name = ui_config.get("window_name", "Face Recognition MVP")
    show_fps = bool(ui_config.get("show_fps", True))

    frame_number = 0
    confirmed_recognitions = 0
    fps_samples: deque[float] = deque(maxlen=60)
    previous_time = time.time()

    print_startup_metrics(references, video_label, runtime_mode, config)
    print("\nPress Q in the video window to exit.\n")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("[WARN] Could not read frame from video source.")
                break

            frame_number += 1

            if frame_number % process_every_n_frames == 0:
                faces = detect_faces(app, frame)
                detections = build_detections(frame, faces, references, config)
                status_events = tracker.update(detections, config)

                for track, previous_status in status_events:
                    became_confirmed = event_logger.maybe_log_status_change(
                        track=track,
                        previous_status=previous_status,
                        frame_number=frame_number,
                    )
                    if became_confirmed:
                        confirmed_recognitions += 1

                    event_logger.maybe_log_failure(
                        track=track,
                        frame_number=frame_number,
                        window_frames=recognition_window_frames,
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

            maybe_print_runtime_metrics(
                frame_number=frame_number,
                active_tracks=len(active_tracks),
                average_fps=average_fps,
                confirmed_recognitions=confirmed_recognitions,
            )

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print("Resources released. Window closed.")


def main() -> None:
    config = load_config()
    run(config)


if __name__ == "__main__":
    main()
