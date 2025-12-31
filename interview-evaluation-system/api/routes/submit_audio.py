from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
import tempfile
import os

from core.models.audio.dsp_stub import BasicDSP
from core.models.audio.asr_stub import FasterWhisperASR

from api.schemas.response_models import (
    TextEvaluationResponse,
    EvaluationBreakdown,
    EvidenceSnippet,
    AudioFeedback
)

router = APIRouter(prefix="/submit_audio", tags=["Evaluation"])

# --------------------------------------------------
# Initialize DSP + ASR once (module-level singletons)
# --------------------------------------------------
dsp = BasicDSP()
asr = FasterWhisperASR(model_size="small", device="cpu", compute_type="int8")


@router.post("/", response_model=TextEvaluationResponse)
def submit_audio_answer(
    http_request: Request,
    question_id: str = Form(...),
    audio_file: UploadFile = File(...)
):
    """
    Evaluate an audio-based interview answer.
    """

    orchestrator = getattr(http_request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(
            status_code=500,
            detail="Evaluation system not initialized"
        )

    if not audio_file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Only WAV audio files are supported"
        )

    # --------------------------------------------------
    # Save uploaded audio temporarily
    # --------------------------------------------------
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.file.read())
            tmp_path = tmp.name

        # --------------------------------------------------
        # DSP → ASR (SINGLE execution)
        # --------------------------------------------------
        audio_signal = dsp.preprocess(tmp_path)
        student_answer = asr.transcribe(audio_signal)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing failed: {str(e)}"
        )

    finally:
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not student_answer:
        raise HTTPException(
            status_code=400,
            detail="Could not transcribe audio"
        )

    # --------------------------------------------------
    # Reuse Day-3 + Day-6 evaluation pipeline
    # --------------------------------------------------
    try:
        result = orchestrator.evaluate(
            question_id=question_id,
            student_answer=student_answer,
            audio_signal=audio_signal  # ✅ Day-6 wire
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
            ],
            audio_feedback=(
                AudioFeedback(**result["audio_feedback"])
                if "audio_feedback" in result
                else None
            )
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )
