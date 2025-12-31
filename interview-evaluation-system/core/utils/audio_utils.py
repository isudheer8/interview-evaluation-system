import numpy as np
import librosa
from typing import Dict


# ============================================================
# Utility functions for audio-based delivery analysis
# India-calibrated, non-punitive
# ============================================================

FRAME_LENGTH = int(0.025 * 16000)  # 25 ms
HOP_LENGTH = int(0.010 * 16000)    # 10 ms


def rms_energy(audio: np.ndarray) -> np.ndarray:
    """
    Compute RMS energy per frame.
    """
    return librosa.feature.rms(
        y=audio,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]


def silence_mask(audio: np.ndarray, threshold_db: float = 20.0) -> np.ndarray:
    """
    Identify silence frames using amplitude thresholding.
    """
    intervals = librosa.effects.split(audio, top_db=threshold_db)
    mask = np.zeros(len(audio), dtype=bool)

    for start, end in intervals:
        mask[start:end] = True

    return mask


# ------------------------------------------------------------
# Metric computations
# ------------------------------------------------------------

def compute_rms_stability(audio: np.ndarray) -> float:
    """
    RMS energy stability (coefficient of variation).
    Lower variation => more stable delivery.
    """
    rms = rms_energy(audio)

    if rms.size == 0 or np.mean(rms) == 0:
        return 0.0

    cv = np.std(rms) / np.mean(rms)

    # India-calibrated normalization (non-punitive)
    if cv < 0.30:
        return 1.0
    elif cv < 0.55:
        return 0.7
    else:
        return 0.4


def compute_pause_ratio(audio: np.ndarray) -> float:
    """
    Ratio of silence duration to total duration.
    """
    total_duration = len(audio) / 16000.0

    silence_intervals = librosa.effects.split(audio, top_db=20)
    speech_duration = sum((end - start) for start, end in silence_intervals) / 16000.0
    silence_duration = max(total_duration - speech_duration, 0.0)

    if total_duration == 0:
        return 0.0

    pause_ratio = silence_duration / total_duration

    # India-aware interpretation
    if pause_ratio <= 0.35:
        return 1.0
    elif pause_ratio <= 0.50:
        return 0.7
    else:
        return 0.4


def compute_speaking_rate(transcript: str, audio: np.ndarray) -> float:
    """
    Words per second (excluding silence).
    """
    words = transcript.strip().split()
    word_count = len(words)

    if word_count == 0:
        return 0.0

    # Estimate speaking time (exclude silence)
    silence_intervals = librosa.effects.split(audio, top_db=20)
    speaking_time = sum((end - start) for start, end in silence_intervals) / 16000.0

    if speaking_time == 0:
        return 0.0

    rate = word_count / speaking_time

    # India-calibrated scoring
    if 1.2 <= rate <= 3.0:
        return 1.0
    elif rate < 1.2 or rate <= 3.5:
        return 0.7
    else:
        return 0.4


# ------------------------------------------------------------
# Aggregated analysis (delivery diagnostics)
# ------------------------------------------------------------

def analyze_audio_delivery(audio: np.ndarray, transcript: str) -> Dict[str, float]:
    """
    Compute delivery-related audio diagnostics.
    This output is meant for feedback, NOT penalties.
    """

    rms_stability = compute_rms_stability(audio)
    pause_score = compute_pause_ratio(audio)
    speaking_rate_score = compute_speaking_rate(transcript, audio)

    delivery_stability = 0.6 * rms_stability + 0.4 * pause_score

    return {
        "delivery_stability": round(delivery_stability, 3),
        "rms_stability": round(rms_stability, 3),
        "pause_score": round(pause_score, 3),
        "speaking_rate_score": round(speaking_rate_score, 3)
    }
