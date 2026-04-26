# Biometric Access Control MVP

Python MVP for real-world face recognition testing with InsightFace, OpenCV, temporal voting, lightweight tracking, quality filtering, RTSP/webcam input, embedding cache, and CSV event logs.

This project is for field evaluation only. It does not open doors, unlock turnstiles, or trigger any real biometric access-control action.

## What This MVP Does

- Reads video from a webcam or RTSP camera, including Intelbras-style RTSP URLs
- Detects faces with InsightFace `buffalo_l`
- Converts BGR to RGB before every model inference
- Keeps BGR frames for OpenCV display
- Filters small and blurry faces before recognition
- Tracks faces across frames with simple center-distance matching
- Uses temporal voting instead of trusting one frame
- Confirms an identity only after enough consistent votes
- Logs recognition events to CSV
- Caches reference embeddings to speed up startup
- Shows bounding boxes, track IDs, scores, votes, and FPS
- Prints runtime metrics in the terminal

Target scale for this MVP: around 10 registered people.

## Project Structure

```text
face_detection/
|-- face_detection.py
|-- config.example.json
|-- config.json                 # local/private, ignored by git
|-- requirements.txt
|-- req.txt
|-- README.md
|-- src/
|   |-- config.py
|   |-- drawing.py
|   |-- logging_utils.py
|   |-- model.py
|   |-- quality.py
|   |-- recognition.py
|   |-- references.py
|   |-- tracking.py
|   `-- video.py
|-- img/
|   `-- references/
|       `-- .gitkeep
|-- data/
|   `-- .gitkeep
`-- logs/
    `-- .gitkeep
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

`req.txt` is kept for compatibility with the earlier project setup. New installs should use `requirements.txt`.

## Configuration

Create your local config from the safe example:

```powershell
Copy-Item config.example.json config.json
```

`config.json` is ignored by git because it may contain camera credentials and local CUDA paths.

### Webcam

```json
"video_source": {
  "type": "webcam",
  "index": 0,
  "width": 1280,
  "height": 720,
  "buffer_size": 1
}
```

### RTSP

```json
"video_source": {
  "type": "rtsp",
  "url": "rtsp://user:password@camera-ip:554/cam/realmonitor?channel=1&subtype=0",
  "buffer_size": 1
}
```

For Intelbras cameras, use the main stream for best face detail:

```text
rtsp://user:password@ip:554/cam/realmonitor?channel=1&subtype=0
```

Use `subtype=1` only if latency or bandwidth is more important than recognition quality.

## Reference Images

Add one folder per person:

```text
img/references/joao_lucas/
    image_1.jpg
    image_2.png

img/references/maria_silva/
    image_1.jpg
```

Folder names become display names:

```text
joao_lucas -> Joao Lucas
maria_silva -> Maria Silva
```

Use clear face images with varied lighting and angles. The loader selects the largest face when multiple faces are detected in a reference image, normalizes each embedding, and stores a normalized mean embedding per person.

## Recognition Stability

Recognition is based on track history, not a single frame.

```json
"recognition": {
  "process_every_n_frames": 2,
  "recognition_window_frames": 12,
  "min_votes_to_confirm": 5,
  "min_average_score": 0.35,
  "candidate_min_score": 0.25,
  "max_unknown_frames": 20
}
```

A track becomes `confirmed` only when the same name appears at least `min_votes_to_confirm` times in the last `recognition_window_frames`, and those votes have an average score greater than or equal to `min_average_score`.

`candidate_min_score` is intentionally lower than the confirmation score so movement and motion blur do not discard every useful frame.

## Face Quality

```json
"quality": {
  "min_face_width": 70,
  "min_face_height": 70,
  "min_blur_score": 25.0,
  "min_detection_score": 0.1
}
```

Faces below the minimum size or blur threshold are ignored. Blur is measured with Laplacian variance on the face crop.

## Tracking

```json
"tracking": {
  "max_center_distance": 180,
  "max_missing_frames": 20
}
```

The tracker is intentionally lightweight. It matches detections to existing tracks by face-center distance, keeps a short prediction history, and removes stale tracks after too many missing frames.

## Cache

```json
"cache": {
  "enabled": true,
  "path": "data/embeddings_cache.pkl",
  "force_rebuild": false
}
```

The cache stores each person name, mean embedding, image count, and a source hash based on reference image paths, sizes, and modification timestamps.

Set `force_rebuild` to `true` after changing reference images if you want to force a fresh embedding build.

## Logging

Events are written to:

```text
logs/recognition_events.csv
```

Columns:

```text
timestamp, track_id, name, status, avg_score, votes, frame_number, x1, y1, x2, y2
```

The logger records status changes, confirmations, and failed unknown attempts after enough frames. It avoids writing a duplicate event on every frame.

## Run

```powershell
python face_detection.py
```

Press `Q` in the video window to exit.

The terminal prints:

- registered people
- video source
- GPU/CPU mode
- active tracks
- average FPS
- confirmed recognitions

## Visualization

- Green box: confirmed
- Yellow box: candidate
- Red box: unknown

Each box displays:

- track ID
- display name
- status
- average score
- vote count

## GPU Check

```powershell
python teste.py
```

If GPU support is available, `CUDAExecutionProvider` should appear. If it does not, the MVP can still run on CPU, but FPS will likely be lower.

## Privacy And Safety

- Do not commit real reference images.
- Do not commit real RTSP credentials.
- Do not use this MVP to make biometric access-control decisions.
- Collect consent from participants before testing.
- Treat logs and embeddings as sensitive biometric-adjacent data.

The `.gitignore` keeps `config.json`, reference images, logs, runtime data, `.venv`, and `__pycache__` out of git while preserving `.gitkeep` files for folder structure.

## Author

[Joao Lucas Oliveira](https://www.linkedin.com/in/joaodosdados/)
