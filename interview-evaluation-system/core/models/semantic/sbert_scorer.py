from sentence_transformers import SentenceTransformer, util
from core.interfaces.semantic_scorer import SemanticScorerInterface

class SBERTSemanticScorer(SemanticScorerInterface):
    """
    Semantic similarity scorer using Sentence-BERT.
    Returns a normalized similarity score in [0, 1].
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def score(self, student_answer: str, ideal_answer: str) -> float:
        if not student_answer or not ideal_answer:
            return 0.0

        emb_student = self.model.encode(
            student_answer, convert_to_tensor=True
        )
        emb_ideal = self.model.encode(
            ideal_answer, convert_to_tensor=True
        )

        similarity = util.cos_sim(emb_student, emb_ideal).item()

        # Normalize cosine similarity from [-1, 1] → [0, 1]
        normalized = (similarity + 1.0) / 2.0
        return max(0.0, min(1.0, normalized))
