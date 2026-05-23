import streamlit as st

from app.prediction_client import friendly_error, predict_diabetes, using_api
from app.ui import configure_page, page_title, result_cards, save_result


configure_page("Medical Assist | Diabetes Prediction")

page_title(
    "Diabetes Model",
    "Diabetes Prediction",
    "Enter patient measurements from the Pima diabetes dataset format. Required biological values are validated before prediction.",
)

st.caption("Prediction mode: FastAPI backend" if using_api() else "Prediction mode: local model files")

with st.form("diabetes_form"):
    st.subheader("Patient Inputs")
    left, right = st.columns(2)

    with left:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucose (mg/dL)", min_value=1.0, max_value=300.0, value=120.0, step=1.0)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=1.0, max_value=200.0, value=72.0, step=1.0)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)

    with right:
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=80.0, step=1.0)
        bmi = st.number_input("BMI", min_value=1.0, max_value=100.0, value=28.0, step=0.1)
        diabetes_pedigree_function = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.45,
            step=0.01,
            format="%.2f",
        )
        age = st.number_input("Age", min_value=1, max_value=120, value=35, step=1)

    submitted = st.form_submit_button("Run Diabetes Prediction")

if submitted:
    payload = {
        "pregnancies": float(pregnancies),
        "glucose": float(glucose),
        "blood_pressure": float(blood_pressure),
        "skin_thickness": float(skin_thickness),
        "insulin": float(insulin),
        "bmi": float(bmi),
        "diabetes_pedigree_function": float(diabetes_pedigree_function),
        "age": int(age),
    }

    try:
        with st.spinner("Analyzing patient measurements..."):
            result = predict_diabetes(payload)

        explanation = (
            "The diabetes probability represents the model's estimated likelihood of the positive diabetes class "
            "for these inputs. A high confidence score means the model strongly favored one class, not that the "
            "prediction is clinically certain."
        )
        save_result("Diabetes Prediction", result, explanation)
        result_cards(result, explanation)

    except Exception as error:
        st.error(friendly_error(error), icon="⚠️")
else:
    st.markdown(
        """
        <div class="card">
            <h3>Input tips</h3>
            <p class="muted">
                Glucose, blood pressure, and BMI must be greater than zero. If a value is unknown,
                use the closest documented value instead of leaving a biological measurement at zero.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
