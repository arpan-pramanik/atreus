"""Unit tests for Concept Drift Detectors."""

import numpy as np
from src.detectors.adwin import ADWINDetector
from src.detectors.ddm import DDMDetector
from src.detectors.eddm import EDDMDetector
from src.detectors.page_hinkley import PageHinkleyDetector


def test_adwin_detection():
    """Verify ADWIN detects sudden error rate jump."""
    detector = ADWINDetector(delta=0.002)
    rng = np.random.RandomState(42)

    # Phase 1: low error (0.05)
    for _ in range(500):
        err = 1 if rng.rand() < 0.05 else 0
        detector.update(err)

    # Phase 2: sudden jump to high error (0.70)
    detected_phase2 = False
    for _ in range(500):
        err = 1 if rng.rand() < 0.70 else 0
        drift, _ = detector.update(err)
        if drift:
            detected_phase2 = True
            break

    assert detected_phase2, "ADWIN should detect concept drift when error rate jumps to 0.70"


def test_ddm_detection():
    """Verify DDM detects error rate degradation."""
    detector = DDMDetector(warm_start=30)
    rng = np.random.RandomState(42)

    # Low error baseline
    for _ in range(200):
        err = 1 if rng.rand() < 0.05 else 0
        detector.update(err)

    # High error degradation
    detected = False
    for _ in range(300):
        err = 1 if rng.rand() < 0.60 else 0
        drift, _ = detector.update(err)
        if drift:
            detected = True
            break

    assert detected, "DDM should detect drift upon sharp increase in error rate"


def test_eddm_detection():
    """Verify EDDM detects concept shift."""
    detector = EDDMDetector(warm_start=30)
    rng = np.random.RandomState(42)

    # Stage 1: rare errors
    for _ in range(500):
        err = 1 if rng.rand() < 0.05 else 0
        detector.update(err)

    # Stage 2: frequent errors
    detected = False
    for _ in range(500):
        err = 1 if rng.rand() < 0.70 else 0
        drift, _ = detector.update(err)
        if drift:
            detected = True
            break

    assert detected or detector.drifts_detected_count > 0


def test_page_hinkley_detection():
    """Verify Page-Hinkley test on stream mean change."""
    detector = PageHinkleyDetector(threshold=10.0, delta=0.005)
    rng = np.random.RandomState(42)

    # Low values
    for _ in range(200):
        val = rng.normal(0.1, 0.02)
        detector.update(val)

    # Elevated values
    detected = False
    for _ in range(200):
        val = rng.normal(1.0, 0.02)
        drift, _ = detector.update(val)
        if drift:
            detected = True
            break

    assert detected, "Page-Hinkley should detect mean shift"
