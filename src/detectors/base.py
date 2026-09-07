"""Base interface for concept drift detection algorithms."""

from abc import ABC, abstractmethod
from typing import Tuple


class BaseDriftDetector(ABC):
    """Abstract Base Class for Concept Drift Detectors."""

    def __init__(self, name: str = "BaseDetector"):
        self.name = name
        self.drift_detected: bool = False
        self.warning_detected: bool = False
        self.total_samples_seen: int = 0
        self.drifts_detected_count: int = 0

    @abstractmethod
    def update(self, val: float) -> Tuple[bool, bool]:
        """
        Process a new sample from the stream.
        
        Args:
            val: Input value (typically error indicator 0/1 or real-valued error/metric).
            
        Returns:
            Tuple[bool, bool]: (drift_detected, warning_detected)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal detector statistics after drift adaptation."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(total_samples={self.total_samples_seen}, drifts={self.drifts_detected_count})"
