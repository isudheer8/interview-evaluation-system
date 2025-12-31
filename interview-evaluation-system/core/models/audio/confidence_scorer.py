from typing import Dict, List


class DeliveryConfidenceScorer:
    """
    Non-punitive delivery confidence & fluency scorer.
    India-calibrated: feedback-first, no negative scoring.
    """

    def __init__(
        self,
        bonus_threshold: float = 0.62,
        max_bonus: float = 0.30
    ):
        """
        Args:
            bonus_threshold: minimum delivery stability to qualify for bonus
            max_bonus: maximum allowed positive bonus
        """
        self.bonus_threshold = bonus_threshold
        self.max_bonus = max_bonus

    def score(self, audio_metrics: Dict[str, float]) -> Dict:
        """
        Generate delivery confidence diagnostics and feedback.

        audio_metrics must contain:
        - delivery_stability
        - rms_stability
        - pause_score
        - speaking_rate_score
        """

        delivery_stability = audio_metrics.get("delivery_stability", 0.0)
        rms_stability = audio_metrics.get("rms_stability", 0.0)
        pause_score = audio_metrics.get("pause_score", 0.0)
        speaking_rate_score = audio_metrics.get("speaking_rate_score", 0.0)

        feedback: List[str] = []

        # -------------------------------
        # Feedback generation (non-judging)
        # -------------------------------
        if delivery_stability >= 0.62:
            feedback.append(
                "Your delivery was stable and confident with clear pacing."
            )
        elif delivery_stability >= 0.40:
            feedback.append(
                "Your delivery was generally clear. Pauses for thinking are normal in technical interviews."
            )
        else:
            feedback.append(
                "Your explanation appears conceptually focused, though the delivery was fragmented. This did not affect evaluation."
            )

        if speaking_rate_score < 0.6:
            feedback.append(
                "You spoke at a careful pace. Taking time to think is acceptable in technical interviews."
            )

        if pause_score < 0.6:
            feedback.append(
                "You took longer pauses while explaining, which is common when recalling technical concepts."
            )

        if rms_stability < 0.6:
            feedback.append(
                "Your voice intensity varied during the answer. Maintaining a steady volume may improve clarity."
            )

        # -------------------------------
        # Bonus logic (positive-only)
        # -------------------------------
        bonus_eligible = delivery_stability >= self.bonus_threshold

        suggested_bonus = (
            round(self.max_bonus * delivery_stability, 2)
            if bonus_eligible
            else 0.0
        )

        return {
            "delivery_stability_score": round(delivery_stability, 3),
            "feedback": feedback,
            "bonus_eligible": bonus_eligible,
            "suggested_bonus": suggested_bonus
        }
