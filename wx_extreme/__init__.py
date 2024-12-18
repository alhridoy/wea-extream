"""WX-Extreme: Advanced Evaluation Framework for Extreme Weather Events in ML Models."""

from wx_extreme.core.detector import ExtremeEventDetector
from wx_extreme.core.evaluator import evaluate_extremes

__version__ = "0.1.0"
__author__ = "Al-Ekram Elahee Hridoy"
__email__ = "aliqramalaheehridoy@gmail.com"

__all__ = [
    "ExtremeEventDetector",
    "evaluate_extremes",
]
