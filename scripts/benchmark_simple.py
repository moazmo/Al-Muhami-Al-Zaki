"""
Simple EgyMMLU Benchmark — Direct LLM Evaluation.

This script directly evaluates the LLM's ability to answer
Egyptian law MCQ questions WITHOUT document retrieval.

This is useful when the knowledge base doesn't match the benchmark questions.

Usage:
    python scripts/benchmark_simple.py
    python scripts/benchmark_simple.py --limit 10
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

from src.utils.logger import setup_logger
from src.utils.config import get_settings


async def answer_mcq_directly(
    question: str,
    choices: List[str],
) -> Dict[str, Any]:
    """
    Use LLM to directly answer an MCQ question.

    Args:
        question: The MCQ question
        choices: List of answer choices

    Returns:
        Dict with predicted_index and answer text
    """
    from langchain_ollama import ChatOllama

    settings = get_settings()

    llm = ChatOllama(
        model=settings.generator_model,
        temperature=0.0,
    )

    # Format choices
    choices_text = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])

    prompt = f"""أنت محامي مصري خبير. أجب على السؤال التالي باختيار الإجابة الصحيحة.

السؤال:
{question}

الخيارات:
{choices_text}

أجب برقم الخيار الصحيح فقط (0, 1, 2, أو 3).
الرقم فقط:"""

    try:
        response = await llm.ainvoke(prompt)
        response_text = response.content.strip()

        # Extract number from response
        predicted = -1
        for char in response_text:
            if char.isdigit():
                predicted = int(char)
                break

        return {
            "predicted_index": predicted,
            "answer_text": response_text,
            "success": True,
        }

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {
            "predicted_index": -1,
            "answer_text": "",
            "success": False,
            "error": str(e),
        }


async def run_simple_benchmark(
    questions: List[Dict[str, Any]],
    output_path: Path,
) -> Dict[str, Any]:
    """
    Run simple LLM-only benchmark.
    """
    logger.info("=" * 60)
    logger.info("Al-Muhami Al-Zaki — Simple LLM Benchmark")
    logger.info("=" * 60)
    logger.info(f"Running {len(questions)} questions (direct LLM, no RAG)")

    results = []

    for i, q in enumerate(questions, 1):
        logger.info(f"[{i}/{len(questions)}] {q['question'][:40]}...")

        start_time = time.time()

        mcq_result = await answer_mcq_directly(
            question=q["question"],
            choices=q["choices"],
        )

        latency = time.time() - start_time

        is_correct = mcq_result["predicted_index"] == q["answer"]

        result = {
            "question": q["question"][:100],
            "subject": q.get("subject", "unknown"),
            "correct_index": q["answer"],
            "predicted_index": mcq_result["predicted_index"],
            "is_correct": is_correct,
            "latency_seconds": round(latency, 2),
            "success": mcq_result["success"],
        }
        results.append(result)

        # Progress update every 5 questions
        if i % 5 == 0:
            correct_so_far = sum(1 for r in results if r.get("is_correct"))
            logger.info(
                f"  📊 Progress: {correct_so_far}/{i} correct ({100 * correct_so_far / i:.1f}%)"
            )

    # Calculate metrics
    successful = [r for r in results if r.get("success")]

    if not successful:
        metrics = {"error": "No successful evaluations"}
    else:
        correct = sum(1 for r in successful if r.get("is_correct"))
        latencies = [r["latency_seconds"] for r in successful]

        metrics = {
            "total_questions": len(results),
            "successful_runs": len(successful),
            "correct": correct,
            "accuracy": round(correct / len(successful), 3),
            "accuracy_pct": round(100 * correct / len(successful), 1),
            "avg_latency_seconds": round(sum(latencies) / len(latencies), 2),
            "total_time_minutes": round(sum(latencies) / 60, 1),
        }

    # Save results
    final_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark": "EgyMMLU_Simple",
        "mode": "Direct LLM (no RAG)",
        "model": get_settings().generator_model,
        "metrics": metrics,
        "detailed_results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to: {output_path}")

    # Print summary
    print_summary(metrics)

    return final_results


def print_summary(metrics: Dict[str, Any]):
    """Print formatted summary."""
    print("\n" + "=" * 60)
    print("📊 SIMPLE LLM BENCHMARK RESULTS")
    print("=" * 60)

    if "error" in metrics:
        print(f"❌ Error: {metrics['error']}")
        return

    print(f"\n📈 Performance Metrics:")
    print(f"  Questions:       {metrics['total_questions']}")
    print(f"  Correct:         {metrics['correct']}/{metrics['successful_runs']}")
    print(f"  Accuracy:        {metrics['accuracy_pct']}%")
    print(f"  Avg Latency:     {metrics['avg_latency_seconds']}s")
    print(f"  Total Time:      {metrics['total_time_minutes']} minutes")

    accuracy = metrics["accuracy_pct"]
    print("\n🎯 GRADE:")
    if accuracy >= 60:
        print("  ⭐⭐⭐ EXCELLENT (≥60%)")
    elif accuracy >= 40:
        print("  ⭐⭐ GOOD (≥40%)")
    elif accuracy >= 25:
        print("  ⭐ FAIR (≥25% = better than random)")
    else:
        print("  ⚠️ BELOW RANDOM (<25%)")

    print("\n📝 README Badge:")
    print(f"  ![EgyMMLU](https://img.shields.io/badge/EgyMMLU-{int(accuracy)}%25-blue)")

    print("\n" + "=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run simple LLM benchmark on EgyMMLU")

    parser.add_argument(
        "--input",
        type=str,
        default="data/eval/egymmlu_law.json",
        help="Path to EgyMMLU questions JSON",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/eval/egymmlu_simple_results.json",
        help="Output path for results",
    )

    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of questions"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    setup_logger(level="INFO")

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Questions file not found: {input_path}")
        return 1

    with open(input_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    logger.info(f"Loaded {len(questions)} questions from {input_path}")

    if args.limit:
        questions = questions[: args.limit]
        logger.info(f"Limited to {len(questions)} questions")

    output_path = Path(args.output)

    try:
        asyncio.run(run_simple_benchmark(questions, output_path))
        return 0
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
