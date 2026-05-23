# src/models/predictors.py
"""
Stateless predictor classes used by both the API and the GUI.
Loading and inference are clearly separated from training code.
"""

import json
import os
from pathlib import Path

import numpy as np
import joblib
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.helpers import get_logger

logger = get_logger("predictors")


class DiabetesPredictor:
    """
    Wraps the sklearn Pipeline (imputer + scaler + classifier).

    The pipeline was saved as a single artifact, so there is no risk of
    applying the wrong scaler to a new model – a common production bug.
    """

    def __init__(self, models_dir: str = "models"):
        pipeline_path = os.path.join(models_dir, "diabetes_pipeline.pkl")
        meta_path     = os.path.join(models_dir, "diabetes_meta.json")

        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(
                f"Diabetes pipeline not found at '{pipeline_path}'. "
                "Run src/training/train_diabetes.py first."
            )

        self.pipeline = joblib.load(pipeline_path)
        self._patch_legacy_sklearn_pipeline()
        logger.info(f"Loaded diabetes pipeline from {pipeline_path}")

        with open(meta_path) as f:
            self.meta = json.load(f)
        self.features: list[str] = self.meta["features"]

    def _patch_legacy_sklearn_pipeline(self) -> None:
        """Add safe defaults for older pickled sklearn estimators."""
        steps = getattr(self.pipeline, "named_steps", {})
        for estimator in steps.values():
            if estimator.__class__.__name__ == "LogisticRegression" and not hasattr(estimator, "multi_class"):
                estimator.multi_class = "auto"

    def predict(self, feature_values: list[float]) -> dict:
        """
        Args:
            feature_values: Values in the SAME order as self.features.

        Returns:
            {
              "prediction": 0 | 1,
              "label": "No Diabetes" | "Diabetes",
              "probability": float,   # P(Diabetes)
              "confidence": float,    # max(P(class))
            }
        """
        if len(feature_values) != len(self.features):
            raise ValueError(
                f"Expected {len(self.features)} features, got {len(feature_values)}."
            )

        X = pd.DataFrame([feature_values], columns=self.features)
        pred  = int(self.pipeline.predict(X)[0])
        proba = float(self.pipeline.predict_proba(X)[0][1])

        return {
            "prediction":  pred,
            "label":       "Diabetes" if pred == 1 else "No Diabetes",
            "probability": round(proba, 4),
            "confidence":  round(max(proba, 1 - proba), 4),
        }


class PneumoniaPredictor:
    """
    Wraps a TensorFlow SavedModel for pneumonia classification.

    Preprocessing (resize + normalise) is done here, not in the GUI/API
    layer, keeping inference logic in one place.
    """

    def __init__(self, models_dir: str = "models"):
        import tensorflow as tf

        self.tf = tf
        self.preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        self._set_float32_policy()

        models_path = Path(models_dir)
        meta_path = models_path / "pneumonia_meta.json"

        self.model_path = self._find_model_path(models_path)
        self.model = self._load_model_safely(self.model_path)
        logger.info("Loaded pneumonia model from %s", self.model_path)

        with open(meta_path) as f:
            self.meta = json.load(f)
        self.img_size: tuple[int, int] = tuple(self.meta.get("img_size", [224, 224]))
        self.class_names: list[str]    = self.meta["class_names"]

    def _set_float32_policy(self) -> None:
        try:
            self.tf.keras.mixed_precision.set_global_policy("float32")
            logger.info("Set TensorFlow mixed precision policy to float32 for inference.")
        except Exception as exc:
            logger.warning("Could not set TensorFlow mixed precision policy: %s", exc)

    def _find_model_path(self, models_path: Path) -> Path:
        candidates = [
            models_path / "pneumonia_model.keras",
            models_path / "pneumonia_model.h5",
            models_path / "pneumonia_model.hdf5",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "Pneumonia model not found. Expected one of: "
            + ", ".join(str(path) for path in candidates)
        )

    def _load_model_safely(self, model_path: Path):
        errors: list[str] = []
        suffix = model_path.suffix.lower()

        if suffix == ".keras":
            load_attempts = [
                {"compile": False, "safe_mode": False},
                {"compile": False},
            ]
        else:
            load_attempts = [{"compile": False}]

        for kwargs in load_attempts:
            try:
                return self.tf.keras.models.load_model(model_path, **kwargs)
            except TypeError as exc:
                errors.append(f"{kwargs}: {exc}")
            except Exception as exc:
                errors.append(f"{kwargs}: {exc}")

        fallback_path = model_path.with_suffix(".h5")
        if suffix == ".keras" and fallback_path.exists():
            try:
                logger.info("Trying pneumonia .h5 fallback at %s", fallback_path)
                return self.tf.keras.models.load_model(fallback_path, compile=False)
            except Exception as exc:
                errors.append(f"{fallback_path}: {exc}")

        joined_errors = " | ".join(errors)
        raise RuntimeError(
            "Pneumonia model could not be loaded. Please re-save the model using "
            "TensorFlow 2.16.2/Keras 3.3.3. Details: "
            f"{joined_errors}"
        )

    def predict_from_path(self, img_path: str) -> dict:
        """Load an image from disk and return prediction."""
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = self.tf.keras.utils.load_img(img_path, target_size=self.img_size, color_mode="rgb")
        return self._predict_img(img)

    def predict_from_bytes(self, img_bytes: bytes) -> dict:
        """Load an image from raw bytes (used by the API)."""
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize(self.img_size)
        return self._predict_img(img)

    def _predict_img(self, img) -> dict:
        """
        Internal helper – accepts a PIL Image or Keras image object.
        EfficientNetB0's built-in preprocessing expects uint8 [0,255].
        """
        arr = self.tf.keras.utils.img_to_array(img).astype(np.float32)
        batch = np.expand_dims(arr, axis=0)                 # 1x224x224x3
        batch = self.preprocess_input(batch)
        score = float(self.model.predict(batch, verbose=0)[0][0])

        # class_names are ordered by directory name (alphabetical):
        # index 0 = NORMAL, index 1 = PNEUMONIA  (matches sigmoid > 0.5)
        pred  = 1 if score >= 0.5 else 0
        label = self.class_names[pred] if self.class_names else ("PNEUMONIA" if pred else "NORMAL")

        return {
            "prediction":  pred,
            "label":       label,
            "probability": round(score, 4),
            "confidence":  round(score if pred == 1 else 1 - score, 4),
        }
