"""DDM (Drift Detection Method) implementation."""

from typing import Tuple
from src.detectors.base import BaseDriftDetector

try:
    from river.drift.binary import DDM as RiverDDM
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False


class DDMDetector(BaseDriftDetector):
    """
    DDM (Drift Detection Method) by Gama et al.
    Monitors online classifier error rate modeled as a Bernoulli trial process.
    """

    def __init__(
        self,
        warm_start: int = 30,
        warning_threshold: float = 2.0,
        drift_threshold: float = 3.0,
    ):
        super().__init__(name="DDM")
        self.warm_start = warm_start
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        self._detector = (
            RiverDDM(
                warm_start=warm_start,
                warning_threshold=warning_threshold,
                drift_threshold=drift_threshold,
            )
            if HAS_RIVER
            else None
        )

    def reset(self) -> None:
        """Reset internal statistics."""
        if HAS_RIVER:
            self._detector = RiverDDM(
                warm_start=self.warm_start,
                warning_threshold=self.warning_threshold,
                drift_threshold=self.drift_threshold,
            )
        self.drift_detected = False
        self.warning_detected = False

    def update(self, val: float) -> Tuple[bool, bool]:
        """
        Update with new classification error (1 for error, 0 for correct).
        
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
