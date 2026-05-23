import streamlit as st

from app.ui import configure_page, disclaimer_box


configure_page("Medical Assist | Home")

st.markdown(
    """
    <div class="hero">
        <span class="pill">AI Healthcare Portfolio Project</span>
        <h1>Medical Assist</h1>
        <p class="subtitle">
            A clean Streamlit interface for diabetes risk prediction and chest X-ray
            pneumonia screening, powered by the project's existing FastAPI and ML models.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
disclaimer_box()
st.write("")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="card">
            <h3>🧪 Diabetes Prediction</h3>
            <p class="muted">Enter eight clinical measurements using a guided form with validation and clear result cards.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="card">
            <h3>🫁 Pneumonia Detection</h3>
            <p class="muted">Upload a JPEG or PNG chest X-ray and review the model prediction, probability, and confidence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="card">
            <h3>📊 Results Summary</h3>
            <p class="muted">View the latest prediction in a polished, mobile-friendly layout suitable for demos and resumes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.subheader("How to use")
st.markdown(
    """
    1. Open a prediction page from the sidebar.
    2. Fill in the form or upload a chest X-ray image.
    3. Click the prediction button and review the result card.
    4. Use the Results page to revisit the latest prediction.
    """
)

st.info(
    "For a deployed FastAPI backend, set MEDICAL_ASSIST_API_URL. Without it, Streamlit uses the local model files in the models folder.",
    icon="ℹ️",
)
