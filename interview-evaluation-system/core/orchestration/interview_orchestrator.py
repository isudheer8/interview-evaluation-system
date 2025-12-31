import re
from typing import Dict, Any, Optional

from core.models.semantic.sbert_scorer import SBERTSemanticScorer
from core.models.keyword.regex_concept_scorer import RegexConceptScorer
from core.models.rag.faiss_retriever import FAISSRetriever
from core.models.fusion.weighted_fusion import WeightedFusionEngine

from core.utils.audio_utils import analyze_audio_delivery
from core.models.audio.confidence_scorer import DeliveryConfidenceScorer
from core.utils.data_loader import QuestionDataLoader
from core.interfaces.orchestrator import InterviewOrchestratorInterface

class InterviewOrchestrator(InterviewOrchestratorInterface):
    """
    Core orchestration engine for interview evaluation.
    Supports text-only and audio-assisted evaluation (feedback-only).
    """

    def __init__(
        self,
        question_data_path: str,
        faiss_index_path: str,
        corpus_chunks: list,
        fusion_weights: Dict[str, float]
    ):
        # ------------------------------
        # Data
        # ------------------------------
        self.data_loader = QuestionDataLoader(question_data_path)
        self.questions = self.data_loader.load()

        # ------------------------------
        # Models
        # ------------------------------
        self.semantic_scorer = SBERTSemanticScorer()
        self.concept_scorer = RegexConceptScorer()
        self.retriever = FAISSRetriever(
            index_path=faiss_index_path,
            corpus=corpus_chunks
        )
        self.fusion_engine = WeightedFusionEngine(fusion_weights)

        # Day-6: Delivery confidence (feedback-only)
        self.delivery_confidence_scorer = DeliveryConfidenceScorer()

        # ------------------------------
        # Index questions by ID
        # ------------------------------
        self.question_map = {
            q.question_id: q for q in self.questions
        }

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------
    def evaluate(
        self,
        question_id: str,
        student_answer: str,
        top_k_evidence: int = 5,
        audio_signal: Optional[Any] = None  # <-- Day-6 addition
    ) -> Dict[str, Any]:

        if question_id not in self.question_map:
            raise ValueError(f"Invalid question_id: {question_id}")

        question = self.question_map[question_id]
        ideal_answer = question.ideal_answers[0]  # MVP: first ideal answer

        # 1️⃣ Semantic Scoring
        semantic_score = self.semantic_scorer.score(
            student_answer,
            ideal_answer.text
        )

        # 2️⃣ Keyword / Concept Scoring
        key_concepts = [
            kc.concept for kc in ideal_answer.key_concepts
        ]
        keyword_score = self.concept_scorer.score(
            student_answer,
            key_concepts
        )

        # 3️⃣ Evidence Retrieval (RAG)
        retrieved_docs = self.retriever.retrieve(
            query=question.question_text + " " + student_answer,
            top_k=top_k_evidence
        )

        evidence_score = 0.0
        if retrieved_docs:
            evidence_score = 0.5  # MVP heuristic


        # 4️⃣ Fusion (TEXT-BASED ONLY)
        scores = {
            "semantic": semantic_score,
            "keyword": keyword_score,
            "evidence": evidence_score
        }

        fused_result = self.fusion_engine.fuse(scores)

        # ------------------------------
        # Base response (text-only safe)
        # ------------------------------
        response = {
            "question_id": question_id,
            "question": question.question_text,
            "student_answer": student_answer,
            "final_score": fused_result["final_score"],
            "verdict": fused_result["verdict"],
            "score_breakdown": scores,
            "evidence_snippets": retrieved_docs[:3]  # limit output
        }

        # --------------------------------------------------
        # Day-6: Audio delivery feedback (OPTIONAL, SAFE)
        # --------------------------------------------------
        if audio_signal is not None:
            audio_metrics = analyze_audio_delivery(
                audio_signal,
                transcript=student_answer
            )

            delivery_feedback = self.delivery_confidence_scorer.score(
                audio_metrics
            )

            response["audio_feedback"] = {
                "delivery_stability_score": delivery_feedback["delivery_stability_score"],
                "feedback": delivery_feedback["feedback"],
                "bonus_eligible": delivery_feedback["bonus_eligible"],
                "suggested_bonus": delivery_feedback["suggested_bonus"]
            }

        return response
