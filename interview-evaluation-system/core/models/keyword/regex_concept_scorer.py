import re
from typing import List
from core.interfaces.concept_scorer import ConceptScorerInterface

class RegexConceptScorer(ConceptScorerInterface):
    """
    Keyword / concept coverage scorer using regex matching.
    """

    def score(self, student_answer: str, key_concepts: List[str]) -> float:
        if not student_answer or not key_concepts:
            return 0.0

        answer = student_answer.lower()
        hits = 0

        for concept in key_concepts:
            pattern = r"\b" + re.escape(concept.lower()) + r"\b"
            if re.search(pattern, answer):
                hits += 1

        return hits / len(key_concepts)
