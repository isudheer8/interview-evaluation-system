from abc import ABC, abstractmethod

class SemanticScorerInterface(ABC):
    """
    Computes semantic similarity between student and ideal answers.
    """

    @abstractmethod
    def score(self, student_answer: str, ideal_answer: str) -> float:
        """
        Output:
            similarity score in range [0, 1]
        """
        pass
