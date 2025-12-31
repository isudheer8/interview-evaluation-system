from dataclasses import dataclass
from typing import List, Dict


@dataclass
class KeyConcept:
    concept: str
    mandatory: bool


@dataclass
class IdealAnswer:
    answer_id: str
    text: str
    key_concepts: List[KeyConcept]
    weight: float


@dataclass
class EvaluationConfig:
    semantic_weight: float
    keyword_weight: float
    evidence_weight: float


@dataclass
class Question:
    question_id: str
    topic: str
    subtopic: str
    difficulty: str
    question_text: str
    ideal_answers: List[IdealAnswer]
    evaluation: EvaluationConfig
    rag_references: Dict
    metadata: Dict
