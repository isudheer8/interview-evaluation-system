import json
from pathlib import Path

from core.orchestration.interview_orchestrator import InterviewOrchestrator

# =====================================================
# PATH SETUP (MATCH REPO STRUCTURE)
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent

QUESTIONS_PATH = (
    BASE_DIR / "data" / "questions" / "questions.json"
)

CORPUS_CHUNKS_PATH = (
    BASE_DIR / "data" / "corpus" / "processed_chunks" / "corpus_chunks.json"
)

FAISS_INDEX_PATH = (
    BASE_DIR / "data" / "embeddings" / "faiss_index.bin"
)

WEIGHTS_PATH = (
    BASE_DIR / "config" / "weights.yaml"
)

# =====================================================
# LOAD HELPERS
# =====================================================
def load_corpus_chunks():
    with open(CORPUS_CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_fusion_weights():
    import yaml
    with open(WEIGHTS_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["fusion_weights"]

# =====================================================
# MAIN DEMO
# =====================================================
def main():
    print("\n==============================")
    print(" Interview Evaluation Demo")
    print("==============================\n")

    corpus_chunks = load_corpus_chunks()
    fusion_weights = load_fusion_weights()

    orchestrator = InterviewOrchestrator(
        question_data_path=str(QUESTIONS_PATH),
        faiss_index_path=str(FAISS_INDEX_PATH),
        corpus_chunks=corpus_chunks,
        fusion_weights=fusion_weights
    )

    # -----------------------------
    # SAMPLE INPUT
    # -----------------------------
    question_id = "ECE_SNS_01"

    student_answer = (
        "The Fourier Transform converts a signal into the frequency domain "
        "so we can analyze its spectral components and understand system behavior."
    )

    # -----------------------------
    # RUN EVALUATION
    # -----------------------------
    result = orchestrator.evaluate(
        question_id=question_id,
        student_answer=student_answer
    )

    # -----------------------------
    # DISPLAY RESULT
    # -----------------------------
    print("Question ID :", result["question_id"])
    print("Question    :", result["question"])
    print("\nStudent Answer:")
    print(result["student_answer"])

    print("\n--- Evaluation Result ---")
    print("Final Score :", result["final_score"], "/ 10")
    print("Verdict     :", result["verdict"])

    print("\nScore Breakdown:")
    for k, v in result["score_breakdown"].items():
        print(f"  {k:10s}: {round(v, 3)}")

    print("\nRetrieved Evidence Snippets:")
    for idx, doc in enumerate(result["evidence_snippets"], start=1):
        print(f"\n[{idx}] ({doc['source_book']}):")
        print(doc["text"][:300], "...")

    print("\n==============================")
    print(" Demo Completed Successfully")
    print("==============================\n")


if __name__ == "__main__":
    main()
