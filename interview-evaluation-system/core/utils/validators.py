class SchemaValidationError(Exception):
    pass


def validate_question_schema(q: dict):
    required_fields = [
        "question_id",
        "topic",
        "subtopic",
        "difficulty",
        "question_text",
        "ideal_answers",
        "evaluation",
        "rag_references",
        "metadata"
    ]

    for field in required_fields:
        if field not in q:
            raise SchemaValidationError(f"Missing required field: {field}")

    if not isinstance(q["ideal_answers"], list) or not q["ideal_answers"]:
        raise SchemaValidationError("ideal_answers must be a non-empty list")

    for ans in q["ideal_answers"]:
        if "text" not in ans or "key_concepts" not in ans:
            raise SchemaValidationError("Invalid ideal_answer structure")

        for kc in ans["key_concepts"]:
            if "concept" not in kc or "mandatory" not in kc:
                raise SchemaValidationError("Invalid key_concept structure")

    eval_cfg = q["evaluation"]
    total_weight = (
        eval_cfg["semantic"]["weight"]
        + eval_cfg["keyword"]["weight"]
        + eval_cfg["evidence"]["weight"]
    )

    if abs(total_weight - 1.0) > 0.01:
        raise SchemaValidationError("Evaluation weights must sum to 1.0")
