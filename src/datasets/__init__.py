"""Datasets package for synthetic and real streaming benchmarks."""

from src.datasets.synthetic import (
    generate_sea_stream,
    generate_agrawal_stream,
    generate_rotating_hyperplane_stream,
)
from src.datasets.loader import load_electricity_stream

__all__ = [
    "generate_sea_stream",
    "generate_agrawal_stream",
    "generate_rotating_hyperplane_stream",
    "load_electricity_stream",
]
