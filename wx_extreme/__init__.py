"""WX-Extreme: Advanced Evaluation Framework for Extreme Weather Events in ML Models."""

from wx_extreme.core.detector import ExtremeEventDetector
from wx_extreme.core.evaluator import evaluate_extremes
from wx_extreme.core.physics import PhysicsValidator
from wx_extreme.core.patterns import PatternAnalyzer
from wx_extreme.core.metrics import (
    ExtremeMetrics,
    PhysicalConsistencyMetrics,
    PatternMetrics,
    ImpactMetrics,
)

__version__ = "0.1.0"
__author__ = "Al-Ekram Elahee Hridoy"
__email__ = "aliqramalaheehridoy@gmail.com"

__all__ = [
    "ExtremeEventDetector",
    "evaluate_extremes",
    "PhysicsValidator",
    "PatternAnalyzer",
    "ExtremeMetrics",
    "PhysicalConsistencyMetrics",
    "PatternMetrics",
    "ImpactMetrics",
]
