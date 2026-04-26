import time
from collections.abc import Generator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.config import load_config
from src.runtime import RecognitionRuntime


runtime: RecognitionRuntime | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runtime
    config = load_config()
    runtime = RecognitionRuntime(config=config, display_window=False)
    runtime.start_background()
    yield
    runtime.stop()


app = FastAPI(title="Biometric Access Control MVP", lifespan=lifespan)
Path("logs/snapshots").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="web/static"), name="static")
app.mount("/snapshots", StaticFiles(directory="logs/snapshots"), name="snapshots")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    with open("web/templates/index.html", "r", encoding="utf-8") as file:
        return file.read()


@app.get("/video")
def video_feed() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/state")
def state() -> dict:
    if runtime is None:
        return {"ready": False}

    snapshot = runtime.snapshot()
    return {
        "ready": True,
        "metrics": snapshot.metrics.__dict__,
        "tracks": snapshot.tracks,
        "events": snapshot.events,
    }


@app.post("/api/stop")
def stop_runtime() -> JSONResponse:
    if runtime is None:
        return JSONResponse({"stopped": False, "message": "Runtime is not initialized."})

    runtime.stop()
    return JSONResponse({"stopped": True, "message": "Camera runtime stopped."})


def _mjpeg_stream() -> Generator[bytes, None, None]:
    while True:
        if runtime is None:
            time.sleep(0.2)
            continue

        if runtime.stop_requested():
            break

        frame = runtime.get_frame_jpeg()
        if frame is None:
            time.sleep(0.2)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.03)
