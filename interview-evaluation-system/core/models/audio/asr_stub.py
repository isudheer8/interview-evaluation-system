import numpy as np
from faster_whisper import WhisperModel

from core.interfaces.asr import ASRInterface


class FasterWhisperASR(ASRInterface):
    """
    Speech-to-text using Faster-Whisper.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8"
    ):
        """
        Args:
            model_size: tiny | base | small | medium
            device: cpu | cuda
            compute_type: int8 | float16 | float32
        """
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

    def transcribe(self, audio_signal: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio to text.
        """

        segments, info = self.model.transcribe(
            audio_signal,
            language="en"
        )

        transcription = []
        for segment in segments:
            transcription.append(segment.text)

        return " ".join(transcription).strip()
