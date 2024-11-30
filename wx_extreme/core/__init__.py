"""Core functionality for wx-extreme package."""

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