from pydantic import BaseModel, Field


class TextEvaluationRequest(BaseModel):
    """
    Request model for text-only interview evaluation.
    """

    question_id: str = Field(
        ...,
        example="ECE_SNS_01",
        description="Unique ID of the interview question"
    )

    student_answer: str = Field(
        ...,
        min_length=5,
        example="The Fourier Transform converts a signal into the frequency domain...",
        description="Student's textual answer to the question"
    )


class ReportRequest(BaseModel):
    """
    Request model for future student reports (placeholder).
    """

    student_id: str = Field(
        ...,
        example="STU_1023",
        description="Unique student identifier"
    )
