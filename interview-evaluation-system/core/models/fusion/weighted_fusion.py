from core.interfaces.fusion_engine import FusionEngineInterface

class WeightedFusionEngine(FusionEngineInterface):
    """
    Combines multiple scores using weighted sum.
    """

    def __init__(self, weights: dict):
        self.weights = weights

    def fuse(self, scores: dict) -> dict:
        final_score = 0.0

        for key, weight in self.weights.items():
            final_score += weight * scores.get(key, 0.0)

        # Scale to 0–10
        final_score = round(final_score * 10, 2)

        verdict = self._map_verdict(final_score)

        return {
            "final_score": final_score,
            "verdict": verdict,
            "breakdown": scores
        }

    def _map_verdict(self, score: float) -> str:
        if score >= 8.0:
            return "Excellent"
        elif score >= 6.0:
            return "Good"
        elif score >= 4.0:
            return "Fair"
        return "Poor"
