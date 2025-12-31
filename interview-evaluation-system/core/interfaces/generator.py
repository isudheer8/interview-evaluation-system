from abc import ABC, abstractmethod
from typing import List

class GeneratorInterface(ABC):
    """
    Generates feedback or explanation using retrieved context.
    """

    @abstractmethod
    def generate(
        self,
        question: str,
        student_answer: str,
        retrieved_docs: List[str]
    ) -> str:
        """
        Output:
            textual feedback / explanation
        """
        pass
