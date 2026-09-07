"""EDDM (Early Drift Detection Method) implementation."""

from typing import Tuple
from src.detectors.base import BaseDriftDetector

try:
    from river.drift.binary import EDDM as RiverEDDM
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False


class EDDMDetector(BaseDriftDetector):
    """
    EDDM (Early Drift Detection Method) by Baena-García et al.
    Monitors distance between consecutive errors to detect gradual and abrupt drift.
    """

    def __init__(
        self,
        warm_start: int = 30,
        alpha: float = 0.95,
        beta: float = 0.90,
    ):
        super().__init__(name="EDDM")
        self.warm_start = warm_start
        self.alpha = alpha
        self.beta = beta
        self._detector = (
            RiverEDDM(
                warm_start=warm_start,
                alpha=alpha,
                beta=beta,
            )
            if HAS_RIVER
            else None
        )

    def reset(self) -> None:
        """Reset internal statistics."""
        if HAS_RIVER:
            self._detector = RiverEDDM(
                warm_start=self.warm_start,
                alpha=self.alpha,
                beta=self.beta,
            )
        self.drift_detected = False
        self.warning_detected = False

    def update(self, val: float) -> Tuple[bool, bool]:
        """
        Update with classification error indicator (1 for error, 0 for correct).
        
        Args:
            val: 1 if error, 0 if correct.
            
        Returns:
            Tuple[bool, bool]: (drift_detected, warning_detected)
        """
        self.total_samples_seen += 1
        self.drift_detected = False
        self.warning_detected = False

        if self._detector is not None:
            self._detector.update(val)
            if self._detector.drift_detected:
                self.drift_detected = True
                self.drifts_detected_count += 1
            if getattr(self._detector, "warning_detected", False):
                self.warning_detected = True

        return self.drift_detected, self.warning_detected
