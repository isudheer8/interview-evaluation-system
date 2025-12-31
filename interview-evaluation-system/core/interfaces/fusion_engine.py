from abc import ABC, abstractmethod
from typing import Any, Dict


class FusionEngineInterface(ABC):
    """
    Combines scores from multiple modalities.
    """

    @abstractmethod
    def fuse(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Input example:
            {
                "semantic": 0.82,
                "keyword": 0.70,
                "audio_confidence": 0.65,
                "evidence": 0.75
            }

        Output:
            {
                "final_score": float,
                "verdict": str,
                "breakdown": Dict[str, float]
            }
        """
        pass
