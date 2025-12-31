from abc import ABC, abstractmethod
from typing import List

class RetrieverInterface(ABC):
    """
    Retrieves supporting knowledge passages.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Output:
            list of retrieved text passages
        """
        pass
