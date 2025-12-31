from abc import ABC, abstractmethod
from typing import List

class ConceptScorerInterface(ABC):
    """
    Detects presence of key concepts in answers.
    """

    @abstractmethod
    def score(self, student_answer: str, key_concepts: List[str]) -> float:
        """
        Output:
            concept coverage score in range [0, 1]
        """
        pass
