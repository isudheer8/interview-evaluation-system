import json
from typing import List
from core.utils.data_models import (
    Question,
    IdealAnswer,
    KeyConcept,
    EvaluationConfig
)
from core.utils.validators import validate_question_schema


class QuestionDataLoader:
    """
    Centralized data loader for interview questions.
    """

    def __init__(self, json_path: str):
        self.json_path = json_path

    def load(self) -> List[Question]:
        with open(self.json_path, "r", encoding="utf-8") as f:
            raw_questions = json.load(f)

        questions = []
        for q in raw_questions:
            validate_question_schema(q)
            questions.append(self._parse_question(q))

        return questions

    def _parse_question(self, q: dict) -> Question:
        ideal_answers = []

        for ans in q["ideal_answers"]:
            key_concepts = [
                KeyConcept(**kc) for kc in ans["key_concepts"]
            ]

            ideal_answers.append(
                IdealAnswer(
                    answer_id=ans["answer_id"],
                    text=ans["text"],
                    key_concepts=key_concepts,
                    weight=ans["weight"]
                )
            )

        eval_cfg = EvaluationConfig(
            semantic_weight=q["evaluation"]["semantic"]["weight"],
            keyword_weight=q["evaluation"]["keyword"]["weight"],
            evidence_weight=q["evaluation"]["evidence"]["weight"]
        )

        return Question(
            question_id=q["question_id"],
            topic=q["topic"],
            subtopic=q["subtopic"],
            difficulty=q["difficulty"],
            question_text=q["question_text"],
            ideal_answers=ideal_answers,
            evaluation=eval_cfg,
            rag_references=q["rag_references"],
            metadata=q["metadata"]
        )
