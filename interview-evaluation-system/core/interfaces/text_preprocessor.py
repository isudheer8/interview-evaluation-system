from abc import ABC, abstractmethod


class TextPreprocessorInterface(ABC):
    """
    Cleans and normalizes interview answers.
    """

    @abstractmethod
    def preprocess(self, text: str) -> str:
        """
        Input:
            raw text
        Output:
            cleaned text (ECE-aware)
        """
        pass
