import numpy as np
import librosa

from core.interfaces.dsp import DSPInterface


class BasicDSP(DSPInterface):
    """
    Basic DSP pipeline for interview audio preprocessing.
    """

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def preprocess(self, audio_path: str) -> np.ndarray:
        """
        Steps:
        1. Load audio
        2. Convert to mono
        3. Resample to 16 kHz
        4. Trim leading/trailing silence
        5. Normalize amplitude
        """

        # Load audio
        audio, sr = librosa.load(
            audio_path,
            sr=self.target_sr,
            mono=True
        )

        if audio.size == 0:
            raise ValueError("Empty audio file")

        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)

        # Normalize amplitude
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        return audio
