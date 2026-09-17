"""
Feature Selection Module — wrapper-based bottleneck feature selection for
deep learning image classifiers.

Available components:
    - FeatureExtractor: extracts bottleneck features from a trained model.
    - FeatureSelector: RFECV-based feature selection with before/after evaluation.
    - Architecture registry: register_architecture, list_available_models, build_model.
"""

from .extractor import (
    FeatureExtractor,
    ARCHITECTURES,
    register_architecture,
    list_available_models,
    build_model,
)
from .wrapper import FeatureSelector

__all__ = [
    "FeatureExtractor",
    "FeatureSelector",
    "ARCHITECTURES",
    "register_architecture",
    "list_available_models",
    "build_model",
]
