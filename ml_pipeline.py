from __future__ import annotations

import json
import hashlib
from collections import Counter
from pathlib import Path
from itertools import product
import random
from datetime import datetime
from typing import Any

import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "House_Room_Dataset"
MODEL_PATH = BASE_DIR / "room_classifier.joblib"
MONITORING_LOG_PATH = BASE_DIR / "prediction_monitoring.jsonl"
IMAGE_SIZE = (160, 160)
CLASS_NAMES = ["Bathroom", "Bedroom", "Dinning", "Kitchen", "Livingroom"]
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_TRAIN_SAMPLES = 2500
DEFAULT_MODEL_TYPE = "ensemble"
FEATURE_VERSION = "hog-color-ensemble-v6"
FEATURE_SELECTION_K = 700
HOG_PARAMS = {
    "orientations": 12,
    "pixels_per_cell": (16, 16),
    "cells_per_block": (2, 2),
    "visualize": False,
    "block_norm": "L2-Hys",
}


def iter_image_paths(dataset_dir: Path = DATASET_DIR):
    for class_name in CLASS_NAMES:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            continue

        for image_path in class_dir.iterdir():
            if image_path.is_file():
                yield class_name, image_path


def load_dataset_overview(dataset_dir: Path = DATASET_DIR) -> dict[str, Any]:
    class_counts: dict[str, int] = {}
    total_images = 0

    for class_name in CLASS_NAMES:
        class_dir = dataset_dir / class_name
        count = 0

        if class_dir.exists():
            count = sum(
                1
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
            )

        class_counts[class_name] = count
        total_images += count

    return {
        "dataset_dir": str(dataset_dir),
        "class_counts": class_counts,
        "total_images": total_images,
        "classes": CLASS_NAMES,
    }


def generate_eda_summary(dataset_dir: Path = DATASET_DIR) -> dict[str, Any]:
    overview = load_dataset_overview(dataset_dir)
    class_counts = overview["class_counts"]
    counts = [count for count in class_counts.values() if count > 0]
    max_count = max(counts) if counts else 0
    min_count = min(counts) if counts else 0
    imbalance_ratio = round(max_count / min_count, 2) if min_count else None
    average_count = round(sum(class_counts.values()) / len(CLASS_NAMES), 2) if CLASS_NAMES else 0
    underrepresented_classes = [
        class_name
        for class_name, count in class_counts.items()
        if average_count and count < average_count * 0.8
    ]

    return {
        "total_images": overview["total_images"],
        "class_counts": class_counts,
        "average_count": average_count,
        "max_count": max_count,
        "min_count": min_count,
        "imbalance_ratio": imbalance_ratio,
        "underrepresented_classes": underrepresented_classes,
        "needs_rebalancing": bool(imbalance_ratio and imbalance_ratio > 1.5),
    }


def clean_dataset(dataset_dir: Path = DATASET_DIR, delete_files: bool = False) -> dict[str, Any]:
    removed_invalid_format = 0
    removed_corrupted = 0
    removed_duplicates = 0
    seen_hashes: set[str] = set()

    for _, image_path in iter_image_paths(dataset_dir):
        if image_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            removed_invalid_format += 1
            if delete_files:
                image_path.unlink(missing_ok=True)
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            removed_corrupted += 1
            if delete_files:
                image_path.unlink(missing_ok=True)
            continue

        file_hash = hashlib.md5(image_path.read_bytes()).hexdigest()
        if file_hash in seen_hashes:
            removed_duplicates += 1
            if delete_files:
                image_path.unlink(missing_ok=True)
            continue

        seen_hashes.add(file_hash)

    return {
        "removed_invalid_format": removed_invalid_format,
        "removed_corrupted": removed_corrupted,
        "removed_duplicates": removed_duplicates,
        "delete_files": delete_files,
    }


def preprocess_image(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, IMAGE_SIZE)
    return resized


def preprocess_gray_image(image: np.ndarray) -> np.ndarray:
    resized = preprocess_image(image)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    normalized = gray.astype("float32") / 255.0
    return normalized


def extract_hog_features(image: np.ndarray) -> np.ndarray:
    preprocessed = preprocess_gray_image(image)
    return hog(preprocessed, **HOG_PARAMS)


def extract_color_features(image: np.ndarray) -> np.ndarray:
    resized = preprocess_image(image)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    rgb_features: list[np.ndarray] = []
    hsv_features: list[np.ndarray] = []
    for channel in cv2.split(rgb):
        hist = cv2.calcHist([channel], [0], None, [16], [0, 256]).flatten()
        rgb_features.append(hist.astype("float32"))

    for channel in cv2.split(hsv):
        hist = cv2.calcHist([channel], [0], None, [16], [0, 256]).flatten()
        hsv_features.append(hist.astype("float32"))

    rgb_moments = np.concatenate(
        [rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1))]
    ).astype(np.float32)
    hsv_moments = np.concatenate(
        [hsv.mean(axis=(0, 1)), hsv.std(axis=(0, 1))]
    ).astype(np.float32)

    color_vector = np.concatenate(
        rgb_features + hsv_features + [rgb_moments, hsv_moments]
    ).astype(np.float32)
    color_sum = color_vector.sum()
    if color_sum > 0:
        color_vector = color_vector / color_sum
    return color_vector


def extract_features(image: np.ndarray) -> np.ndarray:
    hog_features = extract_hog_features(image)
    color_features = extract_color_features(image)
    return np.concatenate([hog_features, color_features]).astype(np.float32)


def build_feature_matrix(
    dataset_dir: Path = DATASET_DIR,
    max_samples: int | None = MAX_TRAIN_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    random_generator = random.Random(42)

    available_classes = [
        class_name for class_name in CLASS_NAMES if (dataset_dir / class_name).exists()
    ]

    if not available_classes:
        raise RuntimeError("No class folders were found in the dataset directory.")

    per_class_limit = None
    if max_samples is not None:
        per_class_limit = max(1, max_samples // len(available_classes))

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            continue

        class_files = [
            image_path
            for image_path in class_dir.iterdir()
            if image_path.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        random_generator.shuffle(class_files)

        if per_class_limit is not None:
            class_files = class_files[:per_class_limit]

        for image_path in class_files:

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            try:
                features.append(extract_features(image))
                labels.append(label)
            except cv2.error:
                continue

    if not features:
        raise RuntimeError("No valid training images were found in the dataset.")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    return X, y


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def balance_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    class_counts = Counter(y_train)
    max_count = max(class_counts.values())
    balanced_features: list[np.ndarray] = []
    balanced_labels: list[np.ndarray] = []

    for class_label in sorted(class_counts):
        class_features = X_train[y_train == class_label]
        class_labels = y_train[y_train == class_label]

        upsampled_features, upsampled_labels = resample(
            class_features,
            class_labels,
            replace=True,
            n_samples=max_count,
            random_state=random_state,
        )

        balanced_features.append(upsampled_features)
        balanced_labels.append(upsampled_labels)

    X_balanced = np.vstack(balanced_features).astype(np.float32)
    y_balanced = np.concatenate(balanced_labels).astype(np.int32)
    return X_balanced, y_balanced


def train_svm_model(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    return train_svm_model_with_params(
        X_train,
        y_train,
        c_value=6,
        gamma_value="scale",
    )


def train_svm_model_with_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    c_value: float = 6,
    gamma_value: str | float = "scale",
) -> Pipeline:
    k_features = min(FEATURE_SELECTION_K, X_train.shape[1])
    model = Pipeline(
        steps=[
            ("feature_selector", SelectKBest(score_func=mutual_info_classif, k=k_features)),
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=c_value,
                    kernel="rbf",
                    gamma=gamma_value,
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_ensemble_model(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    k_features = min(FEATURE_SELECTION_K, X_train.shape[1])
    ensemble_model = Pipeline(
        steps=[
            ("feature_selector", SelectKBest(score_func=mutual_info_classif, k=k_features)),
            ("scaler", StandardScaler()),
            (
                "ensemble",
                VotingClassifier(
                    estimators=[
                        (
                            "svm",
                            SVC(
                                C=6,
                                kernel="rbf",
                                gamma="scale",
                                class_weight="balanced",
                                probability=True,
                                random_state=42,
                            ),
                        ),
                        (
                            "random_forest",
                            RandomForestClassifier(
                                n_estimators=220,
                                max_depth=24,
                                class_weight="balanced_subsample",
                                random_state=42,
                                n_jobs=1,
                            ),
                        ),
                        (
                            "extra_trees",
                            ExtraTreesClassifier(
                                n_estimators=240,
                                max_depth=28,
                                class_weight="balanced",
                                random_state=42,
                                n_jobs=1,
                            ),
                        ),
                    ],
                    voting="soft",
                    n_jobs=1,
                ),
            ),
        ]
    )
    ensemble_model.fit(X_train, y_train)
    return ensemble_model


def evaluate_model(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=CLASS_NAMES,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
    }
    return metrics


def create_model_artifact(model: Any, model_type: str = DEFAULT_MODEL_TYPE) -> dict[str, Any]:
    return {
        "model": model,
        "model_type": model_type,
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "hog_params": HOG_PARAMS,
        "feature_version": FEATURE_VERSION,
    }


def save_model_artifact(
    model: Any,
    model_type: str = DEFAULT_MODEL_TYPE,
    model_path: Path = MODEL_PATH,
) -> Path:
    artifact = create_model_artifact(model, model_type=model_type)
    joblib.dump(artifact, model_path)
    return model_path


def load_model_artifact(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    loaded = joblib.load(model_path)

    if isinstance(loaded, dict) and "model" in loaded:
        return {
            **loaded,
            "class_names": loaded.get("class_names", CLASS_NAMES),
            "image_size": loaded.get("image_size", IMAGE_SIZE),
            "hog_params": loaded.get("hog_params", HOG_PARAMS),
            "feature_version": loaded.get("feature_version"),
            "model_type": loaded.get("model_type", "legacy"),
        }

    return {
        "model": loaded,
        "model_type": "legacy",
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "hog_params": HOG_PARAMS,
        "feature_version": None,
    }


def expected_feature_count() -> int:
    sample_image = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
    return int(extract_features(sample_image).shape[0])


def model_feature_count(model: Any) -> int | None:
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    if hasattr(model, "steps"):
        for _, step in model.steps:
            if hasattr(step, "n_features_in_"):
                return int(step.n_features_in_)
    return None


def is_artifact_compatible(artifact: dict[str, Any]) -> bool:
    model = artifact.get("model")
    if model is None:
        return False

    feature_count = model_feature_count(model)
    if feature_count is not None and feature_count != expected_feature_count():
        return False

    return (
        artifact.get("feature_version") == FEATURE_VERSION
        and artifact.get("model_type") == DEFAULT_MODEL_TYPE
        and tuple(artifact.get("image_size", ())) == IMAGE_SIZE
    )


def train_and_save_model(
    dataset_dir: Path = DATASET_DIR,
    model_path: Path = MODEL_PATH,
    max_samples: int | None = MAX_TRAIN_SAMPLES,
    model_type: str = DEFAULT_MODEL_TYPE,
) -> dict[str, Any]:
    X, y = build_feature_matrix(dataset_dir=dataset_dir, max_samples=max_samples)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    X_train_balanced, y_train_balanced = balance_training_data(X_train, y_train)

    if model_type == "svm":
        model = train_svm_model(X_train_balanced, y_train_balanced)
    elif model_type == "ensemble":
        model = train_ensemble_model(X_train_balanced, y_train_balanced)
    else:
        raise ValueError("Supported model types are 'svm' and 'ensemble'.")

    metrics = evaluate_model(model, X_test, y_test)
    save_model_artifact(model, model_type=model_type, model_path=model_path)

    return {
        "model": model,
        "model_type": model_type,
        "X_shape": X.shape,
        "y_distribution": dict(Counter(y)),
        "balanced_train_distribution": dict(Counter(y_train_balanced)),
        "metrics": metrics,
        "model_path": str(model_path),
    }


def get_or_train_model(
    model_path: Path = MODEL_PATH,
    *,
    train_if_needed: bool = True,
) -> dict[str, Any]:
    if model_path.exists():
        artifact = load_model_artifact(model_path)
        if is_artifact_compatible(artifact):
            return artifact
        if not train_if_needed:
            found_features = model_feature_count(artifact.get("model"))
            raise RuntimeError(
                "Saved model is old or incompatible with the current feature pipeline. "
                f"Expected {expected_feature_count()} features, found "
                f"{found_features if found_features is not None else 'unknown'}. "
                "Retrain the model once to create a compatible artifact."
            )
    elif not train_if_needed:
        raise RuntimeError("No saved model was found. Retrain the model before predicting.")

    result = train_and_save_model(model_path=model_path, model_type=DEFAULT_MODEL_TYPE)
    return create_model_artifact(result["model"])


def predict_image(image_path: str | Path, artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("The uploaded file is not a readable image.")

    if artifact is None:
        artifact = get_or_train_model()

    model = artifact["model"]
    class_names = artifact.get("class_names", CLASS_NAMES)

    features = extract_features(image).reshape(1, -1)
    prediction_index = int(model.predict(features)[0])

    confidence = None
    top_predictions: list[dict[str, float | str]] = []
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities.max()) * 100
        ranked_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = [
            {
                "label": class_names[index],
                "confidence": float(probabilities[index]) * 100,
            }
            for index in ranked_indices
        ]

    return {
        "label": class_names[prediction_index],
        "confidence": confidence,
        "class_index": prediction_index,
        "top_predictions": top_predictions,
    }


def log_prediction_event(
    *,
    predicted_label: str,
    confidence: float | None,
    top_predictions: list[dict[str, float | str]],
    source_filename: str,
    model_type: str,
    log_path: Path = MONITORING_LOG_PATH,
) -> None:
    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "predicted_label": predicted_label,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "top_predictions": top_predictions,
        "source_filename": source_filename,
        "model_type": model_type,
    }
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record) + "\n")


def load_monitoring_summary(log_path: Path = MONITORING_LOG_PATH) -> dict[str, Any]:
    if not log_path.exists():
        return {
            "total_predictions": 0,
            "average_confidence": None,
            "latest_prediction_time": None,
            "class_distribution": {class_name: 0 for class_name in CLASS_NAMES},
            "low_confidence_count": 0,
        }

    records: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    confidence_values = [
        float(record["confidence"])
        for record in records
        if record.get("confidence") is not None
    ]
    class_counter = Counter(record.get("predicted_label") for record in records if record.get("predicted_label"))

    return {
        "total_predictions": len(records),
        "average_confidence": round(sum(confidence_values) / len(confidence_values), 2)
        if confidence_values
        else None,
        "latest_prediction_time": records[-1]["timestamp"] if records else None,
        "class_distribution": {
            class_name: class_counter.get(class_name, 0) for class_name in CLASS_NAMES
        },
        "low_confidence_count": sum(
            1 for value in confidence_values if value < 60
        ),
    }


def evaluate_svm_configuration(
    *,
    dataset_dir: Path = DATASET_DIR,
    max_samples: int | None = MAX_TRAIN_SAMPLES,
    image_size: tuple[int, int] = IMAGE_SIZE,
    orientations: int = HOG_PARAMS["orientations"],
    pixels_per_cell: tuple[int, int] = HOG_PARAMS["pixels_per_cell"],
    cells_per_block: tuple[int, int] = HOG_PARAMS["cells_per_block"],
    c_value: float = 6,
    gamma_value: str | float = "scale",
) -> dict[str, Any]:
    global IMAGE_SIZE, HOG_PARAMS

    old_image_size = IMAGE_SIZE
    old_hog_params = HOG_PARAMS.copy()

    IMAGE_SIZE = image_size
    HOG_PARAMS = {
        **HOG_PARAMS,
        "orientations": orientations,
        "pixels_per_cell": pixels_per_cell,
        "cells_per_block": cells_per_block,
    }

    try:
        X, y = build_feature_matrix(dataset_dir=dataset_dir, max_samples=max_samples)
        X_train, X_test, y_train, y_test = split_dataset(X, y)
        X_train_balanced, y_train_balanced = balance_training_data(X_train, y_train)
        model = train_svm_model_with_params(
            X_train_balanced,
            y_train_balanced,
            c_value=c_value,
            gamma_value=gamma_value,
        )
        metrics = evaluate_model(model, X_test, y_test)
        return {
            "config": {
                "image_size": image_size,
                "orientations": orientations,
                "pixels_per_cell": pixels_per_cell,
                "cells_per_block": cells_per_block,
                "C": c_value,
                "gamma": gamma_value,
                "max_samples": max_samples,
            },
            "metrics": metrics,
        }
    finally:
        IMAGE_SIZE = old_image_size
        HOG_PARAMS = old_hog_params


def run_svm_tuning_grid(
    *,
    dataset_dir: Path = DATASET_DIR,
    max_samples: int | None = 1000,
    image_sizes: list[tuple[int, int]] | None = None,
    orientations_list: list[int] | None = None,
    pixels_per_cell_list: list[tuple[int, int]] | None = None,
    cells_per_block_list: list[tuple[int, int]] | None = None,
    c_values: list[float] | None = None,
    gamma_values: list[str | float] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if image_sizes is None:
        image_sizes = [(128, 128), (160, 160)]
    if orientations_list is None:
        orientations_list = [9, 12, 16]
    if pixels_per_cell_list is None:
        pixels_per_cell_list = [(8, 8), (16, 16)]
    if cells_per_block_list is None:
        cells_per_block_list = [(2, 2)]
    if c_values is None:
        c_values = [3, 6, 10]
    if gamma_values is None:
        gamma_values = ["scale", 0.01]

    results: list[dict[str, Any]] = []

    for (
        image_size,
        orientations,
        pixels_per_cell,
        cells_per_block,
        c_value,
        gamma_value,
    ) in product(
        image_sizes,
        orientations_list,
        pixels_per_cell_list,
        cells_per_block_list,
        c_values,
        gamma_values,
    ):
        result = evaluate_svm_configuration(
            dataset_dir=dataset_dir,
            max_samples=max_samples,
            image_size=image_size,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            c_value=c_value,
            gamma_value=gamma_value,
        )
        results.append(result)

    results.sort(key=lambda item: item["metrics"]["accuracy"], reverse=True)
    return results[:top_k]
