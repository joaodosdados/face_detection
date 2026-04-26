import hashlib
import pickle
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.recognition import ReferenceIdentity, l2_normalize


IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp"}


def normalize_person_name(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").title()


def _iter_reference_images(person_dir: Path) -> list[Path]:
    return sorted(
        file
        for file in person_dir.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def _source_hash(reference_dir: Path) -> str:
    digest = hashlib.sha256()

    for image_path in sorted(reference_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        stat = image_path.stat()
        digest.update(str(image_path.relative_to(reference_dir)).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))

    return digest.hexdigest()


def _face_area(face: Any) -> float:
    x1, y1, x2, y2 = map(float, face.bbox)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _largest_face(faces: list[Any]) -> Any | None:
    if not faces:
        return None
    return max(faces, key=_face_area)


def _read_image_rgb(image_path: Path) -> np.ndarray | None:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"[WARN] OpenCV could not read image: {image_path}")
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _extract_embedding(app: Any, image_path: Path) -> np.ndarray | None:
    image_rgb = _read_image_rgb(image_path)
    if image_rgb is None:
        return None

    faces = app.get(image_rgb)
    face = _largest_face(faces)

    if face is None:
        print(f"[WARN] No face detected in reference image: {image_path}")
        return None

    if len(faces) > 1:
        print(f"[WARN] Multiple faces in {image_path}; using largest face.")

    return l2_normalize(face.embedding)


def _load_cache(cache_path: Path, source_hash: str) -> list[ReferenceIdentity] | None:
    if not cache_path.exists():
        return None

    try:
        with cache_path.open("rb") as file:
            payload = pickle.load(file)
    except (OSError, pickle.PickleError, EOFError) as error:
        print(f"[WARN] Could not load embedding cache: {error}")
        return None

    if payload.get("source_hash") != source_hash:
        return None

    references = [
        ReferenceIdentity(
            name=item["name"],
            embedding=np.asarray(item["embedding"], dtype=np.float32),
            images_count=int(item["images_count"]),
        )
        for item in payload.get("identities", [])
    ]

    if references:
        print(f"Loaded {len(references)} identities from cache: {cache_path}")
        return references

    return None


def _save_cache(cache_path: Path, source_hash: str, references: list[ReferenceIdentity]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": time.time(),
        "source_hash": source_hash,
        "identities": [
            {
                "name": reference.name,
                "embedding": reference.embedding.astype(np.float32),
                "images_count": reference.images_count,
            }
            for reference in references
        ],
    }

    with cache_path.open("wb") as file:
        pickle.dump(payload, file)

    print(f"Saved embedding cache: {cache_path}")


def load_reference_faces(app: Any, config: dict[str, Any]) -> list[ReferenceIdentity]:
    reference_dir = Path(config["paths"]["reference_dir"])
    cache_config = config.get("cache", {})
    cache_enabled = bool(cache_config.get("enabled", True))
    cache_path = Path(cache_config.get("path", "data/embeddings_cache.pkl"))
    force_rebuild = bool(cache_config.get("force_rebuild", False))

    if not reference_dir.exists():
        raise SystemExit(f"Reference folder not found: {reference_dir}")

    source_hash = _source_hash(reference_dir)

    if cache_enabled and not force_rebuild:
        cached_references = _load_cache(cache_path, source_hash)
        if cached_references is not None:
            _print_reference_summary(cached_references)
            return cached_references

    print("\nLoading reference identities...")
    references: list[ReferenceIdentity] = []

    for person_dir in sorted(reference_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        person_name = normalize_person_name(person_dir.name)
        embeddings = [
            embedding
            for image_path in _iter_reference_images(person_dir)
            if (embedding := _extract_embedding(app, image_path)) is not None
        ]

        if not embeddings:
            print(f"[WARN] No valid reference faces for: {person_name}")
            continue

        mean_embedding = l2_normalize(np.mean(embeddings, axis=0))
        references.append(
            ReferenceIdentity(
                name=person_name,
                embedding=mean_embedding.astype(np.float32),
                images_count=len(embeddings),
            )
        )

    if not references:
        raise SystemExit("No valid reference identities were loaded.")

    if cache_enabled:
        _save_cache(cache_path, source_hash, references)

    _print_reference_summary(references)
    return references


def _print_reference_summary(references: list[ReferenceIdentity]) -> None:
    print("\nRegistered identities:")
    for reference in references:
        print(f"- {reference.name}: {reference.images_count} image(s)")
    print(f"Total registered people: {len(references)}")
