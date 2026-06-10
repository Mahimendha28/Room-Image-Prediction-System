import uuid
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory, url_for, jsonify

from ml_pipeline import (
    ALLOWED_EXTENSIONS,
    CLASS_NAMES,
    DEFAULT_MODEL_TYPE,
    IMAGE_SIZE,
    MAX_TRAIN_SAMPLES,
    generate_eda_summary,
    get_or_train_model,
    load_monitoring_summary,
    load_dataset_overview,
    log_prediction_event,
    predict_image,
    train_and_save_model,
)


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


MODEL_ARTIFACT = None
MODEL_LOAD_ERROR = None
try:
    MODEL_ARTIFACT = get_or_train_model(train_if_needed=False)
except RuntimeError as exc:
    MODEL_LOAD_ERROR = str(exc)

DATASET_OVERVIEW = load_dataset_overview()
EDA_SUMMARY = generate_eda_summary()
RETRAIN_MODEL_TYPE = DEFAULT_MODEL_TYPE


def build_stats() -> dict[str, int | str]:
    return {
        "classes": len(CLASS_NAMES),
        "image_size": f"{IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}",
        "training_cap": MAX_TRAIN_SAMPLES if MAX_TRAIN_SAMPLES is not None else "All",
        "total_images": DATASET_OVERVIEW["total_images"],
        "livingroom_images": DATASET_OVERVIEW["class_counts"]["Livingroom"],
        "model_type": (
            MODEL_ARTIFACT.get("model_type", DEFAULT_MODEL_TYPE).title()
            if MODEL_ARTIFACT
            else "Needs Retrain"
        ),
    }


@app.route("/")
@app.route("/home")
def home():
    return render_template(
        "home.html",
        stats=build_stats(),
        class_names=CLASS_NAMES,
        dataset_overview=DATASET_OVERVIEW,
        eda_summary=EDA_SUMMARY,
        monitoring_summary=load_monitoring_summary(),
        model_error=MODEL_LOAD_ERROR,
    )


@app.route("/about")
def about():
    return render_template(
        "about.html",
        stats=build_stats(),
        class_names=CLASS_NAMES,
        dataset_overview=DATASET_OVERVIEW,
        eda_summary=EDA_SUMMARY,
        monitoring_summary=load_monitoring_summary(),
        model_error=MODEL_LOAD_ERROR,
    )


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    global MODEL_ARTIFACT, MODEL_LOAD_ERROR, DATASET_OVERVIEW, EDA_SUMMARY

    prediction = None
    confidence = None
    top_predictions = []
    image_url = None
    error = None
    success = None

    if request.method == "POST":
        action = request.form.get("action", "predict")

        if action == "retrain":
            try:
                train_and_save_model(model_type=RETRAIN_MODEL_TYPE)
                MODEL_ARTIFACT = get_or_train_model(train_if_needed=False)
                MODEL_LOAD_ERROR = None
                DATASET_OVERVIEW = load_dataset_overview()
                EDA_SUMMARY = generate_eda_summary()
                success = "Model retrained successfully. New dataset images are now included."
            except Exception as exc:
                error = f"Retraining failed: {exc}"
        else:
            file = request.files.get("room_image")

            if file is None or file.filename == "":
                error = "Please choose an image before submitting."
            elif not allowed_file(file.filename):
                error = "Only JPG, JPEG, and PNG files are supported."
            else:
                filename = f"{uuid.uuid4().hex}{Path(file.filename).suffix.lower()}"
                saved_path = UPLOAD_DIR / filename
                file.save(saved_path)
                image_url = url_for("static_upload", filename=filename)

                if MODEL_ARTIFACT is None:
                    error = MODEL_LOAD_ERROR or "Model is not ready. Please retrain it once."
                else:
                    try:
                        result = predict_image(saved_path, artifact=MODEL_ARTIFACT)
                        prediction = result["label"]
                        confidence = result["confidence"]
                        top_predictions = result["top_predictions"]
                        log_prediction_event(
                            predicted_label=prediction,
                            confidence=confidence,
                            top_predictions=top_predictions,
                            source_filename=filename,
                            model_type=MODEL_ARTIFACT.get("model_type", DEFAULT_MODEL_TYPE),
                        )
                    except Exception as exc:
                        error = str(exc)

    stats = build_stats()

    return render_template(
        "index.html",
        class_names=CLASS_NAMES,
        dataset_overview=DATASET_OVERVIEW,
        prediction=prediction,
        confidence=confidence,
        top_predictions=top_predictions,
        image_url=image_url,
        error=error,
        success=success,
        stats=stats,
        monitoring_summary=load_monitoring_summary(),
        model_error=MODEL_LOAD_ERROR,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    global MODEL_ARTIFACT
    
    file = request.files.get("room_image")

    if file is None or file.filename == "":
        return jsonify({"error": "Please choose an image before submitting."}), 400
    elif not allowed_file(file.filename):
        return jsonify({"error": "Only JPG, JPEG, and PNG files are supported."}), 400
    
    filename = f"{uuid.uuid4().hex}{Path(file.filename).suffix.lower()}"
    saved_path = UPLOAD_DIR / filename
    file.save(saved_path)
    image_url = url_for("static_upload", filename=filename)

    if MODEL_ARTIFACT is None:
        return jsonify({
            "error": MODEL_LOAD_ERROR or "Model is not ready. Please retrain it once.",
            "image_url": image_url,
        }), 503

    try:
        result = predict_image(saved_path, artifact=MODEL_ARTIFACT)
        log_prediction_event(
            predicted_label=result["label"],
            confidence=result["confidence"],
            top_predictions=result["top_predictions"],
            source_filename=filename,
            model_type=MODEL_ARTIFACT.get("model_type", DEFAULT_MODEL_TYPE),
        )
        return jsonify({
            "success": True,
            "prediction": result["label"],
            "confidence": result["confidence"],
            "top_predictions": result["top_predictions"],
            "image_url": image_url,
            "monitoring": load_monitoring_summary(),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/uploads/<path:filename>")
def static_upload(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
