"""Page-Hinkley Concept Drift Detector."""

from typing import Tuple
from src.detectors.base import BaseDriftDetector

try:
    from river.drift import PageHinkley as RiverPageHinkley
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False


class PageHinkleyDetector(BaseDriftDetector):
    """
    Page-Hinkley test is a sequential analysis technique for detecting change in the average of a stream.
    """

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.9999,
        mode: str = "both",
    ):
        super().__init__(name="Page-Hinkley")
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.mode = mode
        self._detector = (
            RiverPageHinkley(
                min_instances=min_instances,
                delta=delta,
                threshold=threshold,
                alpha=alpha,
                mode=mode,
            )
            if HAS_RIVER
            else None
        )

    def reset(self) -> None:
        """Reset internal statistics."""
        if HAS_RIVER:
            self._detector = RiverPageHinkley(
                min_instances=self.min_instances,
                delta=self.delta,
                threshold=self.threshold,
                alpha=self.alpha,
                mode=self.mode,
            )
        self.drift_detected = False
        self.warning_detected = False

    def update(self, val: float) -> Tuple[bool, bool]:
        """
        Update with new value.
        
        Args:
            val: Real value or error indicator.
            
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

        return self.drift_detected, self.warning_detected
