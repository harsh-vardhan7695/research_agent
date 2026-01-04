"""
Evaluation Harness for Deep Research Agent

This module tests the agent against a curated dataset and measures:
1. Citation quality - are claims grounded in sources?
2. Answer correctness - does it contain expected concepts?
3. Uncertainty handling - does it acknowledge when evidence is weak?
4. Conflict detection - does it note when sources disagree?
5. Session continuity - does context carry across turns?

Design philosophy:
We're NOT trying to get a single "accuracy" number. Research quality is
multi-dimensional and different questions need different metrics.
The goal is to understand WHERE the agent succeeds and fails.
"""

import os
import sys
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agent import ResearchAgent
from core.session import create_session


class EvaluationMetrics:
    """Container for evaluation metrics"""
    
    def __init__(self):
        self.results = []
        self.summary = {}
    
    def add_result(self, result: dict):
        self.results.append(result)
    
    def compute_summary(self):
        """Compute aggregate metrics"""
        if not self.results:
            return {}
        
        # Group by category
        by_category = {}
        for r in self.results:
            cat = r.get("category", "unknown")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r)
        
        self.summary = {
            "total_questions": len(self.results),
            "by_category": {},
            "overall": {
                "avg_concept_coverage": 0,
                "citation_rate": 0,
                "uncertainty_handling": 0,
                "avg_sources_used": 0,
                "avg_response_time": 0
            }
        }
        
        # Per-category metrics
        for cat, results in by_category.items():
            self.summary["by_category"][cat] = {
                "count": len(results),
                "avg_concept_coverage": sum(r.get("concept_coverage", 0) for r in results) / len(results),
                "has_citations": sum(1 for r in results if r.get("has_citations")) / len(results),
                "avg_sources": sum(r.get("sources_used", 0) for r in results) / len(results)
            }
        
        # Overall metrics
        self.summary["overall"]["avg_concept_coverage"] = (
            sum(r.get("concept_coverage", 0) for r in self.results) / len(self.results)
        )
        self.summary["overall"]["citation_rate"] = (
            sum(1 for r in self.results if r.get("has_citations")) / len(self.results)
        )
        self.summary["overall"]["avg_sources_used"] = (
            sum(r.get("sources_used", 0) for r in self.results) / len(self.results)
        )
        self.summary["overall"]["avg_response_time"] = (
            sum(r.get("response_time", 0) for r in self.results) / len(self.results)
        )
        
        # Uncertainty handling (only for questions that need it)
        uncertainty_questions = [r for r in self.results if r.get("expected_behavior") == "uncertainty"]
        if uncertainty_questions:
            self.summary["overall"]["uncertainty_handling"] = (
                sum(1 for r in uncertainty_questions if r.get("shows_uncertainty")) / len(uncertainty_questions)
            )
        
        # Conflict handling
        conflict_questions = [r for r in self.results if r.get("expected_behavior") == "acknowledge_conflict"]
        if conflict_questions:
            self.summary["overall"]["conflict_handling"] = (
                sum(1 for r in conflict_questions if r.get("acknowledges_conflict")) / len(conflict_questions)
            )
        
        return self.summary


def load_dataset(path: str = None) -> dict:
    """Load evaluation dataset"""
    if path is None:
        path = Path(__file__).parent / "dataset.json"
    
    with open(path, 'r') as f:
        return json.load(f)


def check_concept_coverage(answer: str, expected_concepts: list) -> tuple:
    """
    Check what percentage of expected concepts appear in the answer
    
    Returns:
        (coverage_score, found_concepts, missing_concepts)
    """
    answer_lower = answer.lower()
    found = []
    missing = []
    
    for concept in expected_concepts:
        # Check for concept or close variants
        concept_lower = concept.lower()
        if concept_lower in answer_lower:
            found.append(concept)
        else:
            # Check for partial matches (e.g., "embed" for "embeddings")
            if any(concept_lower[:4] in word for word in answer_lower.split()):
                found.append(concept)
            else:
                missing.append(concept)
    
    coverage = len(found) / len(expected_concepts) if expected_concepts else 1.0
    return coverage, found, missing


def check_citations(answer: str) -> tuple:
    """
    Check if the answer contains citations
    
    Returns:
        (has_citations, num_citations, citation_urls)
    """
    # Look for URL patterns
    url_pattern = r'https?://[^\s\)\]<>]+'
    urls = re.findall(url_pattern, answer)
    
    # Look for markdown link patterns [text](url)
    markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', answer)
    
    # Look for citation markers like [Source], [1], etc.
    citation_markers = re.findall(r'\[(?:Source\s*\d*|[\d]+|\w+\s*—\s*[\w.]+)\]', answer)
    
    has_citations = len(urls) > 0 or len(markdown_links) > 0 or len(citation_markers) > 0
    num_citations = len(urls) + len(markdown_links)
    
    return has_citations, num_citations, urls


def check_uncertainty_expression(answer: str) -> bool:
    """Check if answer appropriately expresses uncertainty"""
    uncertainty_phrases = [
        "not publicly available",
        "cannot predict",
        "uncertain",
        "speculative",
        "not possible to know",
        "don't have information",
        "couldn't find",
        "no reliable",
        "unclear",
        "may vary",
        "difficult to determine",
        "not definitively",
        "based on available information"
    ]
    
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in uncertainty_phrases)


def check_conflict_acknowledgment(answer: str) -> bool:
    """Check if answer acknowledges conflicting viewpoints"""
    conflict_phrases = [
        "different perspectives",
        "some sources",
        "others argue",
        "conflicting",
        "disagree",
        "varying opinions",
        "debate",
        "on one hand",
        "alternatively",
        "however",
        "in contrast",
        "different views",
        "some experts",
        "while others"
    ]
    
    answer_lower = answer.lower()
    matches = sum(1 for phrase in conflict_phrases if phrase in answer_lower)
    # Need at least 2 conflict indicators
    return matches >= 2


def evaluate_single_question(
    agent: ResearchAgent,
    question_data: dict,
    verbose: bool = True
) -> dict:
    """
    Evaluate agent on a single question
    
    Returns:
        Dict with evaluation metrics for this question
    """
    question = question_data["question"]
    q_id = question_data["id"]
    category = question_data["category"]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating: {q_id}")
        print(f"Question: {question}")
        print(f"Category: {category}")
    
    start_time = time.time()
    
    # Run the agent
    try:
        result = agent.research_sync(question, max_sources=4)
        answer = result["answer"]
        turn_data = result["turn_data"]
        success = True
    except Exception as e:
        answer = f"Error: {str(e)}"
        turn_data = {}
        success = False
    
    response_time = time.time() - start_time
    
    if verbose:
        print(f"\nAnswer preview: {answer[:200]}...")
        print(f"Response time: {response_time:.1f}s")
    
    # Compute metrics
    eval_result = {
        "id": q_id,
        "category": category,
        "question": question,
        "answer": answer,
        "success": success,
        "response_time": response_time,
        "sources_used": len(turn_data.get("urls_opened", [])),
        "search_queries": turn_data.get("search_queries", [])
    }
    
    # Check concept coverage (for factual/comparison questions)
    if "expected_concepts" in question_data:
        coverage, found, missing = check_concept_coverage(
            answer, question_data["expected_concepts"]
        )
        eval_result["concept_coverage"] = coverage
        eval_result["concepts_found"] = found
        eval_result["concepts_missing"] = missing
        if verbose:
            print(f"Concept coverage: {coverage:.0%} ({len(found)}/{len(question_data['expected_concepts'])})")
    
    # Check citations
    has_citations, num_citations, urls = check_citations(answer)
    eval_result["has_citations"] = has_citations
    eval_result["num_citations"] = num_citations
    eval_result["citation_urls"] = urls
    if verbose:
        print(f"Has citations: {has_citations} ({num_citations} found)")
    
    # Check uncertainty handling
    if question_data.get("expected_behavior") == "uncertainty":
        eval_result["expected_behavior"] = "uncertainty"
        eval_result["shows_uncertainty"] = check_uncertainty_expression(answer)
        if verbose:
            print(f"Shows uncertainty: {eval_result['shows_uncertainty']}")
    
    # Check conflict acknowledgment
    if question_data.get("expected_behavior") == "acknowledge_conflict":
        eval_result["expected_behavior"] = "acknowledge_conflict"
        eval_result["acknowledges_conflict"] = check_conflict_acknowledgment(answer)
        if verbose:
            print(f"Acknowledges conflict: {eval_result['acknowledges_conflict']}")
    
    return eval_result


def evaluate_multi_turn(
    question_data: dict,
    verbose: bool = True
) -> list:
    """
    Evaluate multi-turn conversation
    
    Tests whether context is maintained across turns
    """
    q1 = question_data["question"]
    q2 = question_data.get("followup", "")
    
    if not q2:
        return []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Multi-turn evaluation: {question_data['id']}")
        print(f"Q1: {q1}")
        print(f"Q2: {q2}")
    
    # Create single session for both turns
    agent = ResearchAgent()
    
    results = []
    
    # First turn
    result1 = agent.research_sync(q1, max_sources=3)
    eval1 = {
        "turn": 1,
        "question": q1,
        "answer": result1["answer"],
        "success": True
    }
    
    if "expected_concepts_q1" in question_data:
        coverage, found, missing = check_concept_coverage(
            result1["answer"],
            question_data["expected_concepts_q1"]
        )
        eval1["concept_coverage"] = coverage
    
    results.append(eval1)
    
    # Second turn (same session)
    result2 = agent.research_sync(q2, max_sources=3)
    eval2 = {
        "turn": 2,
        "question": q2,
        "answer": result2["answer"],
        "success": True,
        "session_id": agent.session_id
    }
    
    if "expected_concepts_q2" in question_data:
        coverage, found, missing = check_concept_coverage(
            result2["answer"],
            question_data["expected_concepts_q2"]
        )
        eval2["concept_coverage"] = coverage
    
    # Check if answer references previous context
    # Simple heuristic: mentions terms from first question
    q1_terms = set(q1.lower().split())
    q2_answer_lower = result2["answer"].lower()
    context_maintained = any(term in q2_answer_lower for term in q1_terms if len(term) > 4)
    eval2["context_maintained"] = context_maintained
    
    results.append(eval2)
    
    if verbose:
        print(f"Turn 1 coverage: {eval1.get('concept_coverage', 'N/A')}")
        print(f"Turn 2 coverage: {eval2.get('concept_coverage', 'N/A')}")
        print(f"Context maintained: {context_maintained}")
    
    return results


def run_evaluation(
    dataset_path: str = None,
    max_questions: int = None,
    categories: list = None,
    verbose: bool = True,
    output_path: str = None
) -> dict:
    """
    Run full evaluation on the dataset
    
    Args:
        dataset_path: Path to evaluation dataset JSON
        max_questions: Maximum questions to evaluate (for quick testing)
        categories: Only evaluate these categories
        verbose: Print progress
        output_path: Where to save results
    
    Returns:
        Full evaluation results dict
    """
    # Load dataset
    dataset = load_dataset(dataset_path)
    questions = dataset["questions"]
    
    # Filter by category if specified
    if categories:
        questions = [q for q in questions if q["category"] in categories]
    
    # Limit questions if specified
    if max_questions:
        questions = questions[:max_questions]
    
    print(f"\n{'#'*60}")
    print(f"# Deep Research Agent Evaluation")
    print(f"# Questions: {len(questions)}")
    print(f"# Started: {datetime.now().isoformat()}")
    print(f"{'#'*60}")
    
    metrics = EvaluationMetrics()
    multi_turn_results = []
    
    for q_data in questions:
        # Handle multi-turn separately
        if q_data["category"] == "multi_turn":
            mt_results = evaluate_multi_turn(q_data, verbose=verbose)
            multi_turn_results.extend(mt_results)
            continue
        
        # Create fresh agent for each question (independent sessions)
        agent = ResearchAgent()
        
        result = evaluate_single_question(agent, q_data, verbose=verbose)
        metrics.add_result(result)
        
        # Rate limit protection
        time.sleep(1)
    
    # Compute summary
    summary = metrics.compute_summary()
    
    # Final report
    print(f"\n{'#'*60}")
    print("# EVALUATION SUMMARY")
    print(f"{'#'*60}\n")
    
    print(f"Total questions evaluated: {summary['total_questions']}")
    print(f"\nOverall Metrics:")
    print(f"  - Average concept coverage: {summary['overall']['avg_concept_coverage']:.1%}")
    print(f"  - Citation rate: {summary['overall']['citation_rate']:.1%}")
    print(f"  - Average sources used: {summary['overall']['avg_sources_used']:.1f}")
    print(f"  - Average response time: {summary['overall']['avg_response_time']:.1f}s")
    
    if 'uncertainty_handling' in summary['overall']:
        print(f"  - Uncertainty handling: {summary['overall']['uncertainty_handling']:.1%}")
    if 'conflict_handling' in summary['overall']:
        print(f"  - Conflict handling: {summary['overall']['conflict_handling']:.1%}")
    
    print(f"\nBy Category:")
    for cat, cat_metrics in summary['by_category'].items():
        print(f"  {cat}:")
        print(f"    - Count: {cat_metrics['count']}")
        print(f"    - Concept coverage: {cat_metrics['avg_concept_coverage']:.1%}")
        print(f"    - Has citations: {cat_metrics['has_citations']:.1%}")
    
    # Compile full results
    full_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(questions),
            "dataset_version": dataset.get("metadata", {}).get("version", "unknown")
        },
        "summary": summary,
        "detailed_results": metrics.results,
        "multi_turn_results": multi_turn_results
    }
    
    # Save results if path specified
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")
    
    return full_results


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Deep Research Agent")
    parser.add_argument("--max", type=int, default=None, help="Max questions to evaluate")
    parser.add_argument("--category", type=str, default=None, help="Only this category")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    
    args = parser.parse_args()
    
    categories = [args.category] if args.category else None
    
    run_evaluation(
        max_questions=args.max,
        categories=categories,
        verbose=not args.quiet,
        output_path=args.output
    )
