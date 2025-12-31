import csv
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import sys

# Calculate the project root: D:\Major Project\coding\interview-evaluation-system
# .parent is evaluation/, .parent.parent is experiments/, .parent.parent.parent is root
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from core.orchestration.interview_orchestrator import InterviewOrchestrator
from core.models.semantic.sbert_scorer import SBERTSemanticScorer
from core.models.keyword.regex_concept_scorer import RegexConceptScorer

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
EVAL_CSV = "data/samples/evaluation_set.csv"

QUESTION_DATA_PATH = "data\questions\questions.json"
FAISS_INDEX_PATH = "data/embeddings/faiss_index.bin"
CORPUS_CHUNKS = []  # Not needed for metric scoring

FUSION_WEIGHTS = {
    "semantic": 0.55,
    "keyword": 0.20,
    "evidence": 0.25
}

# --------------------------------------------------
# Load Orchestrator
# --------------------------------------------------
orchestrator = InterviewOrchestrator(
    question_data_path=QUESTION_DATA_PATH,
    faiss_index_path=FAISS_INDEX_PATH,
    corpus_chunks=CORPUS_CHUNKS,
    fusion_weights=FUSION_WEIGHTS
)

semantic_model = SBERTSemanticScorer()
keyword_model = RegexConceptScorer()

# --------------------------------------------------
# Containers
# --------------------------------------------------
human_scores = []
final_scores = []
semantic_only_scores = []
keyword_only_scores = []

# --------------------------------------------------
# Evaluation Loop
# --------------------------------------------------
with open(EVAL_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        question_id = row["question_id"]
        answer = row["student_answer"]
        human = float(row["human_score"])

        question = orchestrator.question_map[question_id]
        ideal = question.ideal_answers[0]

        # --- Semantic only ---
        semantic_score = semantic_model.score(
            answer, ideal.text
        )
        semantic_only_scores.append(semantic_score * 10)

        # --- Keyword only ---
        key_concepts = [kc.concept for kc in ideal.key_concepts]
        keyword_score = keyword_model.score(
            answer, key_concepts
        )
        keyword_only_scores.append(keyword_score * 10)

        # --- Full system score WITHOUT RAG ---
        scores = {
            "semantic": semantic_score,
            "keyword": keyword_score,
            "evidence": 0.0  # explicitly disabled
        }

        fused = orchestrator.fusion_engine.fuse(scores)
        final_scores.append(fused["final_score"])


        human_scores.append(human)

# --------------------------------------------------
# Metrics
# --------------------------------------------------
def mae(pred, gt):
    return np.mean(np.abs(np.array(pred) - np.array(gt)))

def report(name, preds):
    corr, _ = pearsonr(preds, human_scores)
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Pearson Correlation : {corr:.3f}")
    print(f"MAE                : {mae(preds, human_scores):.2f}")

print("\n=== OFFLINE EVALUATION RESULTS ===")

report("Semantic Only", semantic_only_scores)
report("Keyword Only", keyword_only_scores)
report("Semantic + Keyword (Proposed System)", final_scores)
