from abc import ABC, abstractmethod


class ConfigInterface(ABC):
    """
    Handles execution mode and feature toggles.
    """

    @abstractmethod
    def get_mode(self) -> str:
        """
        Returns:
            'text_only', 'audio_text', 'full_multimodal'
        """
        pass
