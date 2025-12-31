from abc import ABC, abstractmethod


class ASRInterface(ABC):
    """
    Abstract interface for speech-to-text models.
    """

    @abstractmethod
    def transcribe(self, audio_signal, sample_rate: int) -> str:
        """
        Convert audio signal to text.

        Args:
            audio_signal: 1D numpy array (mono audio)
            sample_rate (int): Sampling rate of audio

        Returns:
            str: Transcribed text
        """
        pass
