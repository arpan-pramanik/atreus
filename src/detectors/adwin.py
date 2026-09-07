"""ADWIN (Adaptive Windowing) Concept Drift Detector."""

from typing import Tuple
from src.detectors.base import BaseDriftDetector

try:
    from river.drift import ADWIN as RiverADWIN
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False


class ADWINDetector(BaseDriftDetector):
    """
    ADWIN (ADaptive WINdowing) is an adaptive sliding window algorithm for detecting change.
    It automatically adjusts the window size according to the rate of change of the data.
    """

    def __init__(self, delta: float = 0.002, clock: int = 32, max_buckets: int = 5):
        super().__init__(name="ADWIN")
        self.delta = delta
        self.clock = clock
        self.max_buckets = max_buckets
        self._detector = RiverADWIN(delta=delta, clock=clock, max_buckets=max_buckets) if HAS_RIVER else None

    def update(self, val: float) -> Tuple[bool, bool]:
        """
        Feed a new value into ADWIN.
        
        Args:
            val: Real value or error binary indicator (0 or 1).
            
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

    def reset(self) -> None:
        """Reset ADWIN internal state."""
        if HAS_RIVER:
            self._detector = RiverADWIN(delta=self.delta, clock=self.clock, max_buckets=self.max_buckets)
        self.drift_detected = False
        self.warning_detected = False
