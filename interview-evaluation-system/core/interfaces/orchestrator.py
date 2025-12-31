from abc import ABC, abstractmethod
from typing import Any, Dict


class InterviewOrchestratorInterface(ABC):
    """
    Coordinates full interview evaluation flow.
    """

    @abstractmethod
    def evaluate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Payload can contain:
            - text answer
            - audio path
            - question metadata

        Output:
            final evaluation result
        """
        pass
