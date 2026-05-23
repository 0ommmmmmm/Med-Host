from __future__ import annotations

import streamlit as st


DISCLAIMER = (
    "Please consult with a qualified healthcare professional for an diagnosis and personalized medical advice. "
)


def configure_page(title: str = "Medical Assist") -> None:
    st.set_page_config(
        page_title=title,
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    render_sidebar()


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --primary: #1769e0;
            --primary-dark: #0f4fa8;
            --accent: #12b5cb;
            --ink: #172033;
            --muted: #64748b;
            --line: #dbe7f3;
            --panel: #ffffff;
            --soft: #f5f9ff;
            --success: #0f9f6e;
            --warning: #b45309;
            --danger: #dc2626;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(18, 181, 203, 0.12), transparent 32rem),
                linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebarNav"] {
            display: none;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--ink);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink);
        }

        .hero {
            background: linear-gradient(135deg, #ffffff 0%, #eaf5ff 100%);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 2.2rem;
            box-shadow: 0 20px 60px rgba(23, 105, 224, 0.08);
        }

        .hero h1 {
            font-size: clamp(2.1rem, 5vw, 4.2rem);
            line-height: 1.02;
            margin-bottom: 0.75rem;
        }

        .subtitle {
            color: var(--muted);
            font-size: 1.08rem;
            line-height: 1.65;
            max-width: 760px;
        }

        .card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1.25rem;
            box-shadow: 0 12px 34px rgba(15, 79, 168, 0.07);
            height: 100%;
        }

        .metric-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-left: 6px solid var(--primary);
            border-radius: 18px;
            padding: 1.25rem;
            box-shadow: 0 12px 30px rgba(15, 79, 168, 0.08);
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.88rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            color: var(--ink);
            font-size: 1.8rem;
            font-weight: 800;
            line-height: 1.1;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            border-radius: 999px;
            padding: 0.45rem 0.75rem;
            background: #e8f4ff;
            color: var(--primary-dark);
            border: 1px solid #cbe5ff;
            font-weight: 700;
            font-size: 0.9rem;
        }

        .notice {
            background: #fff8e8;
            border: 1px solid #f6d58b;
            border-radius: 16px;
            color: #7c4a03;
            padding: 1rem 1.1rem;
            line-height: 1.55;
        }

        .ok {
            color: var(--success);
        }

        .risk {
            color: var(--danger);
        }

        .muted {
            color: var(--muted);
        }

        div.stButton > button,
        div.stDownloadButton > button {
            border-radius: 999px;
            border: 0;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            color: #ffffff;
            font-weight: 800;
            padding: 0.7rem 1.2rem;
            box-shadow: 0 10px 24px rgba(23, 105, 224, 0.22);
        }

        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
            color: #ffffff;
            border: 0;
            transform: translateY(-1px);
        }

        [data-testid="stMetricValue"] {
            color: var(--ink);
        }

        @media (max-width: 760px) {
            .hero {
                padding: 1.35rem;
                border-radius: 18px;
            }

            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🩺 Medical Assist")
        st.caption("FastAPI + ML + Streamlit")
        st.page_link("streamlit_app.py", label="Home", icon="🏠")
        st.page_link("pages/1_Diabetes_Prediction.py", label="Diabetes Prediction", icon="🧪")
        st.page_link("pages/2_Pneumonia_Detection.py", label="Pneumonia Detection", icon="🫁")
        st.page_link("pages/3_Results.py", label="Results", icon="📊")
        st.page_link("pages/4_About.py", label="About", icon="ℹ️")
        st.divider()
        st.warning(DISCLAIMER, icon="⚠️")


def disclaimer_box() -> None:
    st.markdown(f'<div class="notice"><strong>Medical disclaimer:</strong> {DISCLAIMER}</div>', unsafe_allow_html=True)


def page_title(kicker: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <span class="pill">{kicker}</span>
        <h1>{title}</h1>
        <p class="subtitle">{body}</p>
        """,
        unsafe_allow_html=True,
    )


def result_cards(result: dict, explanation: str) -> None:
    label = result.get("label", "Unknown")
    probability = float(result.get("probability", 0))
    confidence = float(result.get("confidence", 0))
    probability_percent = probability * 100
    confidence_percent = confidence * 100
    risk_class = "risk" if label.lower() in {"diabetes", "pneumonia"} else "ok"

    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Prediction</div>
                <div class="metric-value {risk_class}">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Model Probability</div>
                <div class="metric-value">{probability_percent:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{confidence_percent:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.progress(min(max(confidence, 0.0), 1.0), text=f"Confidence: {confidence_percent:.1f}%")
    st.markdown(
        f"""
        <div class="card">
            <h3>Explanation</h3>
            <p class="muted">{explanation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    disclaimer_box()


def save_result(kind: str, result: dict, explanation: str) -> None:
    st.session_state["latest_result"] = {
        "kind": kind,
        "result": result,
        "explanation": explanation,
    }
