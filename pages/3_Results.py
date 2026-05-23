import streamlit as st

from app.ui import configure_page, page_title, result_cards


configure_page("Medical Assist | Results")

page_title(
    "Latest Output",
    "Results",
    "Review the most recent prediction from either the diabetes or pneumonia workflow.",
)

latest = st.session_state.get("latest_result")

if not latest:
    st.info("No prediction has been run yet. Choose a prediction page from the sidebar to get started.", icon="ℹ️")
    st.stop()

st.subheader(latest["kind"])
result_cards(latest["result"], latest["explanation"])
