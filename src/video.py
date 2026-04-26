from typing import Any

import cv2


def open_video_source(config: dict[str, Any]) -> tuple[cv2.VideoCapture, str]:
    source_config = config.get("video_source") or _legacy_camera_config(config)
    source_type = source_config.get("type", "webcam")

    if source_type == "rtsp":
        url = source_config["url"]
        capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        source_label = _safe_rtsp_label(url)
    elif source_type == "webcam":
        index = int(source_config.get("index", 0))
        capture = cv2.VideoCapture(index)
        source_label = f"webcam:{index}"
    else:
        raise SystemExit(f"Unsupported video source type: {source_type}")

    if not capture.isOpened():
        raise SystemExit(f"Could not open video source: {source_label}")

    if "width" in source_config:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(source_config["width"]))
    if "height" in source_config:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(source_config["height"]))
    if "buffer_size" in source_config:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, int(source_config["buffer_size"]))

    return capture, source_label


def _legacy_camera_config(config: dict[str, Any]) -> dict[str, Any]:
    camera = config.get("camera", {})
    return {
        "type": "webcam",
        "index": camera.get("index", 0),
        "width": camera.get("width", 640),
        "height": camera.get("height", 480),
    }


def _safe_rtsp_label(url: str) -> str:
    if "@" not in url:
        return "rtsp"
    return "rtsp://" + url.split("@", maxsplit=1)[1]
