"""
RAGAS Evaluation Script for Al-Muhami Al-Zaki.

Evaluates the CRAG pipeline using RAGAS metrics:
- Faithfulness: Does the answer stay true to the context?
- Answer Relevancy: Is the answer relevant to the question?
- Context Precision: Are the retrieved docs relevant?
- Context Recall: Did we retrieve all needed info?

Usage:
    python scripts/evaluate_ragas.py --dataset data/eval/test_questions.json
    python scripts/evaluate_ragas.py --quick  # Run with built-in test set
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger

from src.graph.builder import run_query
from src.utils.logger import setup_logger


# -----------------------------------------------------------------------------
# Built-in Test Questions (Egyptian Law focused)
# -----------------------------------------------------------------------------
BUILTIN_TEST_SET = [
    {
        "question": "ما هي حقوق الملكية في القانون المدني المصري؟",
        "ground_truth": "حق الملكية يخول صاحبه التصرف في ملكه والانتفاع به واستغلاله",
        "category": "property_law",
    },
    {
        "question": "ما هي شروط صحة العقد في القانون المصري؟",
        "ground_truth": "يشترط لصحة العقد الرضا والأهلية والمحل والسبب",
        "category": "contract_law",
    },
    {
        "question": "ما هي مدة التقادم في القانون المدني؟",
        "ground_truth": "تتقادم الدعاوى بمضي خمس عشرة سنة ما لم ينص القانون على مدة أقصر",
        "category": "prescription",
    },
    {
        "question": "ما هي أحكام الوكالة في القانون المدني المصري؟",
        "ground_truth": "الوكالة عقد بمقتضاه يلتزم الوكيل بأن يقوم بعمل قانوني لحساب الموكل",
        "category": "agency",
    },
    {
        "question": "ما هو الفرق بين البيع والهبة؟",
        "ground_truth": "البيع عقد بعوض بينما الهبة عقد تبرع بدون مقابل",
        "category": "contracts",
    },
]


async def run_single_evaluation(
    question: str,
    ground_truth: str,
) -> Dict[str, Any]:
    """
    Run a single query and collect data for RAGAS evaluation.

    Args:
        question: Test question
        ground_truth: Expected answer

    Returns:
        Evaluation data dict
    """
    start_time = time.time()

    try:
        result = await run_query(question)

        answer = result.get("generation", "")
        contexts = [doc.page_content for doc in result.get("graded_documents", [])]

        latency = time.time() - start_time

        return {
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": contexts,
            "latency_seconds": round(latency, 2),
            "retrieved_count": len(result.get("documents", [])),
            "graded_count": len(result.get("graded_documents", [])),
            "rewrite_count": result.get("rewrite_count", 0),
            "success": True,
        }

    except Exception as e:
        logger.error(f"Evaluation failed for: {question[:50]}... - {e}")
        return {
            "question": question,
            "ground_truth": ground_truth,
            "answer": "",
            "contexts": [],
            "latency_seconds": time.time() - start_time,
            "error": str(e),
            "success": False,
        }


def compute_simple_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute simple evaluation metrics without external LLM.

    These are proxy metrics that don't require additional API calls.

    Args:
        results: List of evaluation results

    Returns:
        Metrics dict
    """
    successful = [r for r in results if r.get("success")]

    if not successful:
        return {"error": "No successful evaluations"}

    # Basic metrics
    total = len(results)
    success_rate = len(successful) / total

    # Retrieval metrics
    avg_retrieved = sum(r["retrieved_count"] for r in successful) / len(successful)
    avg_graded = sum(r["graded_count"] for r in successful) / len(successful)

    # Latency
    latencies = [r["latency_seconds"] for r in successful]
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = (
        sorted(latencies)[int(len(latencies) * 0.95)]
        if len(latencies) > 1
        else latencies[0]
    )

    # Answer quality proxies
    answers_with_content = sum(1 for r in successful if len(r["answer"]) > 50)
    answers_with_citations = sum(
        1 for r in successful if "مادة" in r["answer"] or "المادة" in r["answer"]
    )

    # Rewrite rate (lower is better)
    rewrites = sum(r["rewrite_count"] for r in successful)
    rewrite_rate = rewrites / len(successful)

    return {
        "total_questions": total,
        "success_rate": round(success_rate, 3),
        "avg_docs_retrieved": round(avg_retrieved, 2),
        "avg_docs_graded_relevant": round(avg_graded, 2),
        "avg_latency_seconds": round(avg_latency, 2),
        "p95_latency_seconds": round(p95_latency, 2),
        "answers_with_content_rate": round(answers_with_content / len(successful), 3),
        "answers_with_citations_rate": round(
            answers_with_citations / len(successful), 3
        ),
        "avg_rewrite_attempts": round(rewrite_rate, 2),
    }


async def run_ragas_evaluation(
    test_set: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run full RAGAS-style evaluation.

    Args:
        test_set: List of test questions with ground truth
        output_path: Optional path to save results

    Returns:
        Evaluation results with metrics
    """
    logger.info("=" * 60)
    logger.info("Al-Muhami Al-Zaki — RAGAS Evaluation")
    logger.info("=" * 60)
    logger.info(f"Running {len(test_set)} test questions...")

    # Run all evaluations
    results = []
    for i, test_case in enumerate(test_set, 1):
        logger.info(f"[{i}/{len(test_set)}] {test_case['question'][:40]}...")

        result = await run_single_evaluation(
            question=test_case["question"],
            ground_truth=test_case.get("ground_truth", ""),
        )
        result["category"] = test_case.get("category", "general")
        results.append(result)

    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_simple_metrics(results)

    # Try RAGAS evaluation if available
    ragas_metrics = {}
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
        )
        from datasets import Dataset

        # Prepare dataset for RAGAS
        ragas_data = {
            "question": [r["question"] for r in results if r["success"]],
            "answer": [r["answer"] for r in results if r["success"]],
            "contexts": [r["contexts"] for r in results if r["success"]],
            "ground_truth": [r["ground_truth"] for r in results if r["success"]],
        }

        if ragas_data["question"]:
            dataset = Dataset.from_dict(ragas_data)

            logger.info("Running RAGAS evaluation (requires LLM calls)...")
            ragas_result = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                ],
            )
            ragas_metrics = dict(ragas_result)

    except ImportError:
        logger.warning("RAGAS not fully configured - using simple metrics only")
    except Exception as e:
        logger.warning(f"RAGAS evaluation failed: {e}")

    # Combine results
    final_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_set_size": len(test_set),
        "simple_metrics": metrics,
        "ragas_metrics": ragas_metrics,
        "detailed_results": results,
    }

    # Save if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to: {output_path}")

    # Print summary
    print_evaluation_summary(metrics, ragas_metrics)

    return final_results


def print_evaluation_summary(
    simple_metrics: Dict[str, Any],
    ragas_metrics: Dict[str, Any],
):
    """Print a formatted evaluation summary."""
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)

    print("\n📈 Pipeline Metrics:")
    print(f"  Success Rate:        {simple_metrics.get('success_rate', 0):.1%}")
    print(f"  Avg Docs Retrieved:  {simple_metrics.get('avg_docs_retrieved', 0):.1f}")
    print(
        f"  Avg Docs Relevant:   {simple_metrics.get('avg_docs_graded_relevant', 0):.1f}"
    )
    print(f"  Avg Latency:         {simple_metrics.get('avg_latency_seconds', 0):.1f}s")
    print(f"  P95 Latency:         {simple_metrics.get('p95_latency_seconds', 0):.1f}s")

    print("\n📝 Answer Quality:")
    print(
        f"  Has Content:         {simple_metrics.get('answers_with_content_rate', 0):.1%}"
    )
    print(
        f"  Has Citations:       {simple_metrics.get('answers_with_citations_rate', 0):.1%}"
    )
    print(f"  Avg Rewrites:        {simple_metrics.get('avg_rewrite_attempts', 0):.2f}")

    if ragas_metrics:
        print("\n🎯 RAGAS Metrics:")
        for key, value in ragas_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.3f}")

    print("\n" + "=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on Al-Muhami Al-Zaki"
    )

    parser.add_argument(
        "--dataset", type=str, help="Path to JSON file with test questions"
    )

    parser.add_argument(
        "--quick", action="store_true", help="Run with built-in test set (5 questions)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/eval/results.json",
        help="Output path for results",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logger(level="INFO")

    # Load test set
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            return 1

        with open(dataset_path, "r", encoding="utf-8") as f:
            test_set = json.load(f)
    else:
        # Use built-in test set
        logger.info("Using built-in test set (5 questions)")
        test_set = BUILTIN_TEST_SET

    # Run evaluation
    output_path = Path(args.output)

    try:
        asyncio.run(run_ragas_evaluation(test_set, output_path))
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
