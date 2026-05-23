import streamlit as st

from app.ui import configure_page, disclaimer_box, page_title


configure_page("Medical Assist | About")

page_title(
    "Project Notes",
    "About Medical Assist",
    "A beginner-friendly machine learning project that combines model training, FastAPI inference, and a Streamlit demo UI.",
)

disclaimer_box()
st.write("")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="card">
            <h3>Tech stack</h3>
            <p class="muted">
                Streamlit frontend, FastAPI backend, Pydantic validation, scikit-learn pipeline for diabetes,
                TensorFlow/Keras model for pneumonia image classification, Pillow for image handling.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="card">
            <h3>Model workflow</h3>
            <p class="muted">
                The frontend can call FastAPI through MEDICAL_ASSIST_API_URL or use the existing predictor
                classes directly when deployed as a single Streamlit app.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.subheader("Models")
st.markdown(
    """
    - **Diabetes:** scikit-learn pipeline using tabular clinical inputs from the Pima-style feature set.
    - **Pneumonia:** TensorFlow/Keras chest X-ray classifier that predicts `NORMAL` or `PNEUMONIA`.
    - **Inference layer:** `src/models/predictors.py` centralizes preprocessing and prediction logic.
    """
)

st.subheader("Limitations")
st.markdown(
    """
    - The app is not clinically validated and must not be used for real medical decisions.
    - Predictions depend on dataset quality, preprocessing, model calibration, and input quality.
    - Chest X-ray results require radiologist review and patient history.
    - Confidence is a model score, not a guarantee of medical correctness.
    """
)
