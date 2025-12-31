from fastapi import FastAPI
from pathlib import Path
import json
import yaml

from core.orchestration.interview_orchestrator import InterviewOrchestrator
from api.routes.submit_text import router as submit_text_router
from api.routes.submit_audio import router as submit_audio_router

# =====================================================
# PATH SETUP
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent

QUESTIONS_PATH = BASE_DIR / "data" / "questions" / "questions.json"
CORPUS_CHUNKS_PATH = BASE_DIR / "data" / "corpus" / "processed_chunks" / "corpus_chunks.json"
FAISS_INDEX_PATH = BASE_DIR / "data" / "embeddings" / "faiss_index.bin"
WEIGHTS_PATH = BASE_DIR / "config" / "weights.yaml"

# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(
    title="Interview Evaluation System API",
    version="0.1.0"
)

# =====================================================
# STARTUP: LOAD ORCHESTRATOR ONCE
# =====================================================
@app.on_event("startup")
def load_system():
    with open(CORPUS_CHUNKS_PATH, "r", encoding="utf-8") as f:
        corpus_chunks = json.load(f)

    with open(WEIGHTS_PATH, "r") as f:
        weights_cfg = yaml.safe_load(f)

    app.state.orchestrator = InterviewOrchestrator(
        question_data_path=str(QUESTIONS_PATH),
        faiss_index_path=str(FAISS_INDEX_PATH),
        corpus_chunks=corpus_chunks,
        fusion_weights=weights_cfg["fusion_weights"]
    )

    print("[STARTUP] Interview Orchestrator loaded successfully")

# =====================================================
# ROUTES
# =====================================================
app.include_router(submit_text_router)
app.include_router(submit_audio_router)


# =====================================================
# HEALTH CHECK
# =====================================================
@app.get("/health")
def health():
    return {"status": "ok"}
