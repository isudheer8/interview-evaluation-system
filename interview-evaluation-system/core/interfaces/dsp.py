from abc import ABC, abstractmethod
import numpy as np


class DSPInterface(ABC):
    """
    Abstract interface for audio preprocessing.
    """

    @abstractmethod
    def preprocess(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio.

        Args:
            audio_path (str): Path to input WAV file

        Returns:
            np.ndarray: Cleaned mono audio signal at 16 kHz
        """
        pass
