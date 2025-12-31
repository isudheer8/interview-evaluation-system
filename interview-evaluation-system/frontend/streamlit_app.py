import streamlit as st
import requests

# -----------------------------
# Configuration
# -----------------------------
API_BASE_URL = "http://127.0.0.1:8000"
SUBMIT_TEXT_URL = f"{API_BASE_URL}/submit_text/"
SUBMIT_AUDIO_URL = f"{API_BASE_URL}/submit_audio/"

st.set_page_config(
    page_title="Interview Evaluation System",
    layout="centered"
)

# -----------------------------
# Header
# -----------------------------
st.title("🎓 Interview Evaluation System")
st.caption("Hybrid SLM + RAG based Technical Interview Evaluation (ECE)")

st.divider()

# -----------------------------
# Question Selection
# -----------------------------
st.subheader("Select Question")

# MVP: Hardcoded for demo (can be loaded from CSV later)
QUESTION_MAP = {
    "ECE_SNS_01": "Explain the significance of the Fourier Transform in signal analysis.",
    "ECE_AE_01": "Explain the operation of a BJT in active region.",
    "ECE_CS_01": "What is the role of feedback in control systems?"
}

question_id = st.selectbox(
    "Choose a question",
    options=list(QUESTION_MAP.keys()),
    format_func=lambda x: QUESTION_MAP[x]
)

question_text = QUESTION_MAP[question_id]
st.info(question_text)

st.divider()

# -----------------------------
# Answer Input Tabs
# -----------------------------
tab_text, tab_audio = st.tabs(["📝 Text Answer", "🎙️ Audio Answer"])

# TEXT INPUT
with tab_text:
    student_text = st.text_area(
        "Enter your answer",
        height=180,
        placeholder="Type your technical answer here..."
    )
    submit_text = st.button("Evaluate Text Answer")

# AUDIO INPUT
with tab_audio:
    audio_file = st.file_uploader(
        "Upload WAV audio file",
        type=["wav"]
    )
    submit_audio = st.button("Evaluate Audio Answer")

st.divider()

# -----------------------------
# Helper: Render Results
# -----------------------------
def render_result(result: dict):
    st.subheader("📊 Evaluation Result")

    st.metric(
        label="Final Score",
        value=f"{result['final_score']} / 10",
        delta=result["verdict"]
    )

    st.markdown("### Score Breakdown")
    st.progress(result["score_breakdown"]["semantic"])
    st.write("Semantic Score")

    st.progress(result["score_breakdown"]["keyword"])
    st.write("Keyword Score")

    st.progress(result["score_breakdown"]["evidence"])
    st.write("Evidence Score")

    # -------------------------
    # Audio Feedback (Optional)
    # -------------------------
    if result.get("audio_feedback"):
        st.markdown("### 🎧 Delivery Feedback")

        af = result["audio_feedback"]
        st.write(
            f"**Delivery Stability Score:** {af['delivery_stability_score']}"
        )

        for fb in af["feedback"]:
            st.write(f"• {fb}")

    # -------------------------
    # Evidence Snippets
    # -------------------------
    st.markdown("### 📚 Retrieved Evidence")

    for idx, doc in enumerate(result["evidence_snippets"], start=1):
        with st.expander(f"Evidence {idx} — {doc['source_book']}"):
            st.write(f"**Domain:** {doc['domain']}")
            st.write(doc["text"])


# -----------------------------
# Handle Text Submission
# -----------------------------
if submit_text:
    if not student_text.strip():
        st.warning("Please enter a text answer.")
    else:
        with st.spinner("Evaluating answer..."):
            payload = {
                "question_id": question_id,
                "student_answer": student_text
            }
            response = requests.post(SUBMIT_TEXT_URL, json=payload)

        if response.status_code == 200:
            render_result(response.json())
        else:
            st.error(response.text)

# -----------------------------
# Handle Audio Submission
# -----------------------------
if submit_audio:
    if audio_file is None:
        st.warning("Please upload a WAV audio file.")
    else:
        with st.spinner("Processing audio..."):
            files = {
                "audio_file": (audio_file.name, audio_file, "audio/wav")
            }
            data = {
                "question_id": question_id
            }
            response = requests.post(
                SUBMIT_AUDIO_URL,
                files=files,
                data=data
            )

        if response.status_code == 200:
            render_result(response.json())
        else:
            st.error(response.text)
