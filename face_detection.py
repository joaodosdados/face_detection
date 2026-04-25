import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CONFIG_PATH = "config.json"


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        print(f"Erro: arquivo de configuração não encontrado: {config_path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


CONFIG = load_config(CONFIG_PATH)


def setup_cuda_paths(config: dict[str, Any]) -> None:
    cuda_path = config["paths"].get("cuda_path")
    cudnn_path = config["paths"].get("cudnn_path")

    for dll_path in [cuda_path, cudnn_path]:
        if dll_path and os.path.exists(dll_path):
            os.add_dll_directory(dll_path)
            os.environ["PATH"] = dll_path + os.pathsep + os.environ["PATH"]


setup_cuda_paths(CONFIG)

import insightface  # noqa: E402


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def normalize_person_name(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").title()


def load_model(config: dict[str, Any]):
    model_cfg = config["model"]

    model_name = model_cfg["name"]
    use_gpu = model_cfg["use_gpu"]
    det_size = tuple(model_cfg["det_size"])
    detection_threshold = model_cfg.get("detection_threshold")

    if use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1

    print("Carregando modelo...")

    try:
        app = insightface.app.FaceAnalysis(
            name=model_name,
            providers=providers,
        )
        app.prepare(ctx_id=ctx_id, det_size=det_size)

        if detection_threshold is not None:
            detector = app.models.get("detection")
            if detector is not None:
                detector.det_thresh = detection_threshold
                print(f"Detection threshold ajustado para: {detection_threshold}")

        print("Modelo carregado com GPU." if use_gpu else "Modelo carregado com CPU.")
        return app

    except Exception as error:
        print(f"[AVISO] Falha ao carregar GPU. Usando CPU. Erro: {error}")

        app = insightface.app.FaceAnalysis(
            name=model_name,
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=det_size)

        if detection_threshold is not None:
            detector = app.models.get("detection")
            if detector is not None:
                detector.det_thresh = detection_threshold

        print("Modelo carregado com CPU.")
        return app
    
def load_reference_faces(app, config: dict[str, Any]) -> list[dict[str, Any]]:
    reference_dir = Path(config["paths"]["reference_dir"])

    if not reference_dir.exists():
        print(f"Erro: pasta de referências não encontrada: {reference_dir}")
        sys.exit(1)

    references = []

    print("\nCarregando referências...")

    for person_dir in sorted(reference_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        person_name = normalize_person_name(person_dir.name)
        embeddings = []

        image_files = sorted([
            file
            for file in person_dir.iterdir()
            if file.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
        ])

        if not image_files:
            print(f"[AVISO] Nenhuma imagem encontrada para: {person_name}")
            continue

        for image_path in image_files:
            image = cv2.imread(str(image_path))

            print(f"\n[DEBUG] Lendo imagem: {image_path}")
            print(f"[DEBUG] Caminho absoluto: {image_path.resolve()}")
            print(f"[DEBUG] Existe? {image_path.exists()}")

            if image is None:
                print(f"[ERRO] OpenCV não conseguiu carregar: {image_path}")
                continue

            print(f"[DEBUG] Shape: {image.shape}")

            # 🔥 CORREÇÃO CRÍTICA
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            faces = app.get(image_rgb)

            print(f"[DEBUG] Faces detectadas: {len(faces)}")

            if len(faces) == 0:
                print(f"[AVISO] Nenhuma face detectada em: {image_path}")
                continue

            if len(faces) > 1:
                print(f"[AVISO] Mais de uma face em {image_path}. Usando a primeira.")

            embeddings.append(faces[0].embedding)

        if not embeddings:
            print(f"[AVISO] Nenhuma face válida para: {person_name}")
            continue

        mean_embedding = np.mean(embeddings, axis=0)

        references.append({
            "name": person_name,
            "embedding": mean_embedding,
            "images_count": len(embeddings),
        })

        print(f"OK: {person_name} ({len(embeddings)} imagens)")

    if not references:
        print("Erro: nenhuma referência válida foi carregada.")
        sys.exit(1)

    print(f"\nTotal de pessoas carregadas: {len(references)}")
    return references


def identify_face(
    face_embedding: np.ndarray,
    references: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[str, float, bool]:
    threshold = config["recognition"]["threshold"]
    min_margin = config["recognition"]["margin"]
    soft_threshold = config["recognition"].get("soft_threshold", threshold - 0.08)

    scores = []

    for reference in references:
        score = cosine_similarity(reference["embedding"], face_embedding)
        scores.append((reference["name"], score))

    scores.sort(key=lambda item: item[1], reverse=True)

    best_name, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else -1.0
    margin = best_score - second_score

    if best_score >= threshold and (len(scores) == 1 or margin >= min_margin):
        return best_name, best_score, True

    if best_score >= soft_threshold and margin >= 0:
        return f"{best_name}?", best_score, True

    return "Desconhecido", best_score, False

def draw_face(frame, face, name: str, score: float, matched: bool) -> None:
    x1, y1, x2, y2 = map(int, face.bbox)

    color = (0, 255, 0) if matched else (0, 0, 255)
    label = f"{name} ({score:.2f})"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    y_text = max(y1 - 10, 25)

    cv2.putText(
        frame,
        label,
        (x1, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def draw_fps(frame, fps: float) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )


def open_camera(config: dict[str, Any]):
    camera_cfg = config["camera"]

    cap = cv2.VideoCapture(camera_cfg["index"])

    if not cap.isOpened():
        print("Erro: webcam não encontrada.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg["height"])

    return cap


def main() -> None:
    app = load_model(CONFIG)
    references = load_reference_faces(app, CONFIG)
    cap = open_camera(CONFIG)

    process_every_n_frames = CONFIG["recognition"]["process_every_n_frames"]
    show_fps = CONFIG["ui"]["show_fps"]
    window_name = CONFIG["ui"]["window_name"]

    frame_count = 0
    last_faces = []
    prev_time = time.time()
    fps = 0.0

    print("\nWebcam iniciada.")
    print("Pressione Q para sair.\n")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Erro ao capturar frame da webcam.")
                break

            frame_count += 1

            if frame_count % process_every_n_frames == 0:
                last_faces = app.get(frame)

            for face in last_faces:
                name, score, matched = identify_face(
                    face.embedding,
                    references,
                    CONFIG,
                )
                draw_face(frame, face, name, score, matched)

            current_time = time.time()
            elapsed = current_time - prev_time

            if elapsed > 0:
                fps = 1 / elapsed

            prev_time = current_time

            if show_fps:
                draw_fps(frame, fps)

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Recursos liberados. Janela fechada.")


if __name__ == "__main__":
    main()