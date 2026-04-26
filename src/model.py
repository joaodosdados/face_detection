from typing import Any

import cv2
import numpy as np


def _configure_detection_threshold(app: Any, detection_threshold: float | None) -> None:
    if detection_threshold is None:
        return

    detector = app.models.get("detection")
    if detector is None:
        print("[WARN] Detection model not found; threshold was not changed.")
        return

    detector.det_thresh = detection_threshold
    print(f"Detection threshold: {detection_threshold}")


def _create_face_analysis(config: dict[str, Any], providers: list[str], ctx_id: int) -> Any:
    import insightface

    model_config = config["model"]
    app = insightface.app.FaceAnalysis(
        name=model_config["name"],
        providers=providers,
    )
    app.prepare(ctx_id=ctx_id, det_size=tuple(model_config["det_size"]))
    _configure_detection_threshold(app, model_config.get("detection_threshold"))

    return app


def load_model(config: dict[str, Any]) -> tuple[Any, str]:
    model_config = config["model"]
    use_gpu = bool(model_config.get("use_gpu", False))

    print("Loading InsightFace model...")

    if use_gpu:
        try:
            app = _create_face_analysis(
                config=config,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                ctx_id=0,
            )
            return app, "GPU"
        except Exception as error:
            print(f"[WARN] GPU initialization failed. Falling back to CPU. Error: {error}")

    app = _create_face_analysis(
        config=config,
        providers=["CPUExecutionProvider"],
        ctx_id=-1,
    )
    return app, "CPU"


def detect_faces(app: Any, frame_bgr: np.ndarray) -> list[Any]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return app.get(frame_rgb)
