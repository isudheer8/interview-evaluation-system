from typing import Dict, List, Any, Optional
from pydantic import BaseModel


class EvidenceSnippet(BaseModel):
    source_book: str
    authors: str
    domain: str
    text: str


class EvaluationBreakdown(BaseModel):
    semantic: float
    keyword: float
    evidence: float


# --------------------------------------------------
# Day-6: Audio feedback schema (optional)
# --------------------------------------------------
class AudioFeedback(BaseModel):
    delivery_stability_score: float
    feedback: List[str]
    bonus_eligible: bool
    suggested_bonus: float


class TextEvaluationResponse(BaseModel):
    """
    Unified response model for text and audio evaluation.
    """

    question_id: str
    question: str
    student_answer: str

    final_score: float
    verdict: str

    score_breakdown: EvaluationBreakdown
    evidence_snippets: List[EvidenceSnippet]

    # Day-6 addition (optional, audio-only)
    audio_feedback: Optional[AudioFeedback] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Any = None
