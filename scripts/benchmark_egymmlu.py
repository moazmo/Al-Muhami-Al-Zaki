"""
EgyMMLU Benchmark Script for Al-Muhami Al-Zaki.

Runs the CRAG pipeline against EgyMMLU law questions
and calculates accuracy and faithfulness metrics.

Usage:
    python scripts/benchmark_egymmlu.py
    python scripts/benchmark_egymmlu.py --limit 10  # Quick test
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger

from src.graph.builder import run_query
from src.utils.logger import setup_logger
from src.utils.config import get_settings


async def evaluate_mcq_answer(
    question: str,
    choices: List[str],
    generated_answer: str,
    correct_index: int,
) -> Dict[str, Any]:
    """
    Evaluate if the generated answer matches the correct MCQ choice.

    Uses simple text matching and keyword extraction.

    Args:
        question: The original question
        choices: List of MCQ choices
        generated_answer: Our system's answer
        correct_index: Index of correct choice

    Returns:
        Evaluation result with predicted and correct answers
    """
    from langchain_ollama import ChatOllama

    settings = get_settings()

    # Use Ollama (local, unlimited) for MCQ evaluation
    llm = ChatOllama(
        model=settings.grader_model,
        temperature=0.0,
    )

    # Format choices
    choices_text = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])

    prompt = f"""أنت مقيّم إجابات. بناءً على الإجابة المقدمة، حدد أي خيار من الخيارات التالية يتطابق معها أكثر.

السؤال: {question}

الخيارات:
{choices_text}

الإجابة المقدمة:
{generated_answer}

أجب برقم الخيار فقط (0, 1, 2, أو 3). إذا لم تتطابق الإجابة مع أي خيار، أجب بـ -1.
الرقم فقط:"""

    try:
        response = await llm.ainvoke(prompt)
        predicted_str = response.content.strip()

        # Extract number from response
        predicted = -1
        for char in predicted_str:
            if char.isdigit():
                predicted = int(char)
                break

        is_correct = predicted == correct_index

        return {
            "predicted_index": predicted,
            "correct_index": correct_index,
            "is_correct": is_correct,
            "predicted_choice": choices[predicted]
            if 0 <= predicted < len(choices)
            else "N/A",
            "correct_choice": choices[correct_index],
        }

    except Exception as e:
        logger.error(f"MCQ evaluation failed: {e}")
        return {
            "predicted_index": -1,
            "correct_index": correct_index,
            "is_correct": False,
            "error": str(e),
        }


async def run_single_benchmark(
    question: str,
    choices: List[str],
    correct_index: int,
    subject: str,
) -> Dict[str, Any]:
    """
    Run benchmark on a single EgyMMLU question.

    Args:
        question: MCQ question in Arabic
        choices: List of answer choices
        correct_index: Index of correct answer
        subject: Question subject/category

    Returns:
        Benchmark result dict
    """
    start_time = time.time()

    try:
        # Run CRAG pipeline
        result = await run_query(question)

        answer = result.get("generation", "")
        graded_docs = result.get("graded_documents", [])
        rewrite_count = result.get("rewrite_count", 0)

        latency = time.time() - start_time

        # Check if answer cites sources (faithfulness proxy)
        has_citation = any(
            keyword in answer
            for keyword in ["مادة", "المادة", "القانون", "الدستور", "1948", "2014"]
        )

        # Evaluate MCQ correctness
        mcq_result = await evaluate_mcq_answer(question, choices, answer, correct_index)

        return {
            "question": question,
            "subject": subject,
            "generated_answer": answer,
            "choices": choices,
            "correct_index": correct_index,
            "predicted_index": mcq_result["predicted_index"],
            "is_correct": mcq_result["is_correct"],
            "has_citation": has_citation,
            "docs_retrieved": len(result.get("documents", [])),
            "docs_relevant": len(graded_docs),
            "rewrite_count": rewrite_count,
            "latency_seconds": round(latency, 2),
            "success": True,
        }

    except Exception as e:
        logger.error(f"Benchmark failed for question: {e}")
        return {
            "question": question,
            "subject": subject,
            "is_correct": False,
            "has_citation": False,
            "latency_seconds": time.time() - start_time,
            "error": str(e),
            "success": False,
        }


async def run_egymmlu_benchmark(
    questions: List[Dict[str, Any]],
    output_path: Path,
) -> Dict[str, Any]:
    """
    Run full EgyMMLU benchmark.

    Args:
        questions: List of EgyMMLU questions
        output_path: Path to save results

    Returns:
        Benchmark results with metrics
    """
    logger.info("=" * 60)
    logger.info("Al-Muhami Al-Zaki — EgyMMLU Benchmark")
    logger.info("=" * 60)
    logger.info(f"Running {len(questions)} questions...")
    logger.info("⏱️ Rate limiting enabled: 4s delay between questions")

    results = []

    for i, q in enumerate(questions, 1):
        logger.info(f"[{i}/{len(questions)}] {q['question'][:40]}...")

        result = await run_single_benchmark(
            question=q["question"],
            choices=q["choices"],
            correct_index=q["answer"],
            subject=q.get("subject", "unknown"),
        )
        results.append(result)

        # Progress update every 5 questions
        if i % 5 == 0:
            correct_so_far = sum(1 for r in results if r.get("is_correct"))
            logger.info(
                f"  📊 Progress: {correct_so_far}/{i} correct ({100 * correct_so_far / i:.1f}%)"
            )

        # Rate limiting: wait 4 seconds between questions to avoid API limits
        if i < len(questions):
            await asyncio.sleep(4)

    # Calculate metrics
    successful = [r for r in results if r.get("success")]
    total = len(results)

    if not successful:
        metrics = {"error": "No successful evaluations"}
    else:
        correct = sum(1 for r in successful if r.get("is_correct"))
        with_citations = sum(1 for r in successful if r.get("has_citation"))
        with_relevant_docs = sum(1 for r in successful if r.get("docs_relevant", 0) > 0)

        latencies = [r["latency_seconds"] for r in successful]

        metrics = {
            "total_questions": total,
            "successful_runs": len(successful),
            "accuracy": round(correct / len(successful), 3),
            "accuracy_pct": round(100 * correct / len(successful), 1),
            "faithfulness": round(with_citations / len(successful), 3),
            "faithfulness_pct": round(100 * with_citations / len(successful), 1),
            "retrieval_hit_rate": round(with_relevant_docs / len(successful), 3),
            "avg_latency_seconds": round(sum(latencies) / len(latencies), 2),
            "total_time_minutes": round(sum(latencies) / 60, 1),
        }

    # Save results
    final_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark": "EgyMMLU",
        "metrics": metrics,
        "detailed_results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to: {output_path}")

    # Print summary
    print_benchmark_summary(metrics)

    return final_results


def print_benchmark_summary(metrics: Dict[str, Any]):
    """Print formatted benchmark summary."""
    print("\n" + "=" * 60)
    print("📊 EgyMMLU BENCHMARK RESULTS")
    print("=" * 60)

    if "error" in metrics:
        print(f"❌ Error: {metrics['error']}")
        return

    print(f"\n📈 Performance Metrics:")
    print(f"  Questions:       {metrics['total_questions']}")
    print(f"  Accuracy:        {metrics['accuracy_pct']}%")
    print(f"  Faithfulness:    {metrics['faithfulness_pct']}%")
    print(f"  Retrieval Rate:  {metrics['retrieval_hit_rate'] * 100:.1f}%")
    print(f"  Avg Latency:     {metrics['avg_latency_seconds']}s")
    print(f"  Total Time:      {metrics['total_time_minutes']} minutes")

    # Grade
    accuracy = metrics["accuracy_pct"]
    print("\n🎯 GRADE:")
    if accuracy >= 80:
        print("  ⭐⭐⭐ EXCELLENT (≥80%)")
    elif accuracy >= 60:
        print("  ⭐⭐ GOOD (≥60%)")
    elif accuracy >= 40:
        print("  ⭐ FAIR (≥40%)")
    else:
        print("  ⚠️ NEEDS IMPROVEMENT (<40%)")

    print("\n📝 README Badge:")
    print(
        f"  ![EgyMMLU](https://img.shields.io/badge/EgyMMLU-{int(accuracy)}%25_Accuracy-green)"
    )

    print("\n" + "=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run EgyMMLU benchmark on Al-Muhami Al-Zaki"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/eval/egymmlu_law.json",
        help="Path to EgyMMLU questions JSON",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/eval/egymmlu_results.json",
        help="Output path for results",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions (for quick testing)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logger(level="INFO")

    # Load questions
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Questions file not found: {input_path}")
        logger.info("Run 'python download_egymmlu.py' first to download the dataset.")
        return 1

    with open(input_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    logger.info(f"Loaded {len(questions)} questions from {input_path}")

    # Apply limit if specified
    if args.limit:
        questions = questions[: args.limit]
        logger.info(f"Limited to {len(questions)} questions")

    # Run benchmark
    output_path = Path(args.output)

    try:
        asyncio.run(run_egymmlu_benchmark(questions, output_path))
        return 0
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
