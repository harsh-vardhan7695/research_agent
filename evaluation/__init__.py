"""
Evaluation module for Deep Research Agent

Contains:
- dataset.json: Test questions across categories
- evaluator.py: Evaluation harness and metrics
"""

from .evaluator import (
    run_evaluation,
    evaluate_single_question,
    EvaluationMetrics,
    load_dataset
)

__all__ = [
    "run_evaluation",
    "evaluate_single_question", 
    "EvaluationMetrics",
    "load_dataset"
]
