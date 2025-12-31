from fastapi import APIRouter, HTTPException, Request

from api.schemas.request_models import TextEvaluationRequest
from api.schemas.response_models import (
    TextEvaluationResponse,
    EvaluationBreakdown,
    EvidenceSnippet
)

router = APIRouter(prefix="/submit_text", tags=["Evaluation"])


@router.post("/", response_model=TextEvaluationResponse)
def submit_text_answer(
    request: TextEvaluationRequest,
    http_request: Request
):
    """
    Evaluate a text-based interview answer.
    """

    orchestrator = getattr(http_request.app.state, "orchestrator", None)

    if orchestrator is None:
        raise HTTPException(
            status_code=500,
            detail="Evaluation system not initialized"
        )

    try:
        result = orchestrator.evaluate(
            question_id=request.question_id,
            student_answer=request.student_answer
        )

        return TextEvaluationResponse(
            question_id=result["question_id"],
            question=result["question"],
            student_answer=result["student_answer"],
            final_score=result["final_score"],
            verdict=result["verdict"],
            score_breakdown=EvaluationBreakdown(
                semantic=result["score_breakdown"]["semantic"],
                keyword=result["score_breakdown"]["keyword"],
                evidence=result["score_breakdown"]["evidence"]
            ),
            evidence_snippets=[
                EvidenceSnippet(
                    source_book=doc["source_book"],
                    authors=doc["authors"],
                    domain=doc["domain"],
                    text=doc["text"]
                )
                for doc in result["evidence_snippets"]
            ]
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal evaluation error: {str(e)}"
        )
