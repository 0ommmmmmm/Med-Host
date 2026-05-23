from PIL import Image
import streamlit as st

from app.prediction_client import friendly_error, predict_pneumonia, using_api
from app.ui import configure_page, page_title, result_cards, save_result


configure_page("Medical Assist | Pneumonia Detection")

page_title(
    "Chest X-ray Model",
    "Pneumonia Detection",
    "Upload a frontal chest X-ray image. The model returns a NORMAL or PNEUMONIA prediction with confidence details.",
)

st.caption("Prediction mode: FastAPI backend" if using_api() else "Prediction mode: local model files")

uploaded_file = st.file_uploader(
    "Upload chest X-ray image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

if uploaded_file:
    image_bytes = uploaded_file.getvalue()

    if len(image_bytes) > 10 * 1024 * 1024:
        st.error("File is too large. Please upload an image under 10 MB.", icon="⚠️")
        st.stop()

    preview_col, action_col = st.columns([1, 1])
    with preview_col:
        st.image(Image.open(uploaded_file).convert("RGB"), caption="Uploaded chest X-ray", width="stretch")

    with action_col:
        st.markdown(
            """
            <div class="card">
                <h3>Ready to analyze</h3>
                <p class="muted">
                    The image will be resized and preprocessed by the existing pneumonia predictor before inference.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        run_prediction = st.button("Run Pneumonia Detection")

    if run_prediction:
        try:
            with st.spinner("Reviewing chest X-ray image..."):
                result = predict_pneumonia(image_bytes, uploaded_file.name, uploaded_file.type)

            explanation = (
                "The probability is the model's estimated score for the pneumonia class. X-ray interpretation "
                "requires clinical context and radiology review, so this output should be treated as an educational demo."
            )
            save_result("Pneumonia Detection", result, explanation)
            result_cards(result, explanation)

        except Exception as error:
            st.error(friendly_error(error), icon="⚠️")
            st.caption("Diabetes prediction and the rest of the app are still available.")
else:
    st.markdown(
        """
        <div class="card">
            <h3>Upload guidance</h3>
            <p class="muted">
                Use a clear JPEG or PNG chest X-ray image under 10 MB. Non-image files and oversized uploads are blocked.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
