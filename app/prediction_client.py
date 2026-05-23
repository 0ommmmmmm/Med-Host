from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st


API_URL = os.getenv("MEDICAL_ASSIST_API_URL", "").rstrip("/")
MODELS_DIR = Path("models")


@st.cache_resource(show_spinner=False)
def get_diabetes_predictor():
    from src.models.predictors import DiabetesPredictor

    return DiabetesPredictor(str(MODELS_DIR))


@st.cache_resource(show_spinner=False)
def get_pneumonia_predictor():
    from src.models.predictors import PneumoniaPredictor

    return PneumoniaPredictor(str(MODELS_DIR))


def using_api() -> bool:
    return bool(API_URL)


def predict_diabetes(payload: dict[str, Any]) -> dict:
    if using_api():
        response = requests.post(f"{API_URL}/predict/diabetes", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    features = [
        payload["pregnancies"],
        payload["glucose"],
        payload["blood_pressure"],
        payload["skin_thickness"],
        payload["insulin"],
        payload["bmi"],
        payload["diabetes_pedigree_function"],
        payload["age"],
    ]
    return get_diabetes_predictor().predict(features)


def predict_pneumonia(image_bytes: bytes, filename: str, content_type: str) -> dict:
    if using_api():
        files = {"file": (filename, image_bytes, content_type)}
        response = requests.post(f"{API_URL}/predict/pneumonia", files=files, timeout=120)
        response.raise_for_status()
        return response.json()

    try:
        return get_pneumonia_predictor().predict_from_bytes(image_bytes)
    except RuntimeError:
        get_pneumonia_predictor.clear()
        raise


def friendly_error(error: Exception) -> str:
    if isinstance(error, requests.HTTPError) and error.response is not None:
        try:
            detail = error.response.json().get("detail", error.response.text)
        except ValueError:
            detail = error.response.text
        return f"Prediction service error: {detail}"

    if isinstance(error, requests.RequestException):
        return "Could not reach the FastAPI backend. Check MEDICAL_ASSIST_API_URL or run the API locally."

    message = str(error)
    if "Pneumonia model could not be loaded" in message or "deserialized properly" in message:
        return (
            "Pneumonia model could not be loaded. Please re-save the model using "
            "TensorFlow 2.16.2/Keras 3.3.3."
        )

    return str(error)
