#!/usr/bin/env python3
"""
Custom Egyptian Law Benchmark Runner.

Runs the CRAG pipeline against questions specifically designed
to match the ingested Egyptian law documents.

Usage:
    python scripts/benchmark_egyptian.py [--limit N]
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.graph.builder import run_query
from src.utils.config import get_settings

# Configure logging
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    level="INFO",
)


def load_benchmark(path: str = "data/eval/egyptian_law_benchmark.json") -> List[Dict]:
    """Load the custom Egyptian law benchmark questions."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def format_question(q: Dict) -> str:
    """Format a question for the CRAG pipeline."""
    question_text = q["question"]
    choices = q["choices"]

    formatted = f"{question_text}\n\n"
    for key, value in choices.items():
        formatted += f"{key}) {value}\n"

    return formatted


def extract_answer_choice(generated: str, choices: Dict[str, str]) -> str:
    """
    Extract the answer choice (A/B/C/D) from generated text using regex patterns.

    Looks for patterns like:
    - "A)" or "B)" at start of lines
    - "الإجابة: A" or "الإجابة الصحيحة: B"
    - "الإجابة هي: C" or "الجواب: D"
    - Bold markers like "**A)**" or "**B**"
    """
    import re

    # Normalize text
    text = generated.strip()

    # Priority patterns (most explicit mentions of the answer)
    priority_patterns = [
        # Arabic explicit answer patterns
        r"الإجابة\s*(?:الصحيحة)?\s*(?:هي)?[:：]?\s*([A-Da-d])",
        r"الجواب\s*(?:الصحيح)?\s*(?:هو)?[:：]?\s*([A-Da-d])",
        r"الاختيار\s*(?:الصحيح)?[:：]?\s*([A-Da-d])",
        # Bold answer patterns
        r"\*\*([A-Da-d])\)",
        r"\*\*([A-Da-d])\*\*\)",
        # "Answer: X" pattern
        r"(?:^|\n)\s*([A-Da-d])\)",
    ]

    for pattern in priority_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    # Secondary: Look for choice content matching
    # Check if the answer text contains the exact choice content
    for letter, choice_text in choices.items():
        # Check if the choice text appears prominently
        if choice_text in text:
            # Make sure it's indicated as the answer (not just mentioned)
            choice_pattern = rf"{letter}\)\s*{re.escape(choice_text[:20])}"
            if re.search(choice_pattern, text):
                return letter

    # Tertiary: Count letter mentions in answer context
    letter_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for letter in letter_counts:
        # Count explicit choice mentions
        pattern = rf"(?:^|\s){letter}\)|\({letter}\)|{letter}[)）]"
        letter_counts[letter] = len(re.findall(pattern, text, re.MULTILINE))

    # Return the most mentioned letter if any has clear majority
    max_count = max(letter_counts.values())
    if max_count > 0:
        for letter, count in letter_counts.items():
            if count == max_count:
                return letter

    return "NONE"


async def evaluate_answer(
    generated: str, correct_answer: str, choices: Dict[str, str]
) -> Dict[str, Any]:
    """
    Evaluate if the generated answer matches the correct choice.
    Uses regex-based extraction (faster and more reliable than LLM).

    Returns dict with:
        - correct: bool
        - extracted_answer: str (A/B/C/D or NONE)
        - correct_answer: str
    """
    extracted = extract_answer_choice(generated, choices)
    is_correct = extracted == correct_answer

    return {
        "correct": is_correct,
        "extracted_answer": extracted,
        "correct_answer": correct_answer,
    }


async def run_benchmark(
    questions: List[Dict],
    limit: int = None,
    output_path: str = "data/eval/egyptian_law_results.json",
) -> Dict[str, Any]:
    """Run the full benchmark."""

    if limit:
        questions = questions[:limit]

    results = []
    correct_count = 0
    total_time = 0

    logger.info(f"Starting Egyptian Law Benchmark with {len(questions)} questions")

    for i, q in enumerate(questions, 1):
        start_time = datetime.now()

        # Format question for CRAG
        formatted_q = format_question(q)

        logger.info(f"[{i}/{len(questions)}] {q['question'][:50]}...")

        try:
            # Run through CRAG pipeline
            result = await run_query(formatted_q)
            generated = result.get("generation", "")

            # Evaluate answer
            eval_result = await evaluate_answer(
                generated, q["correct_answer"], q["choices"]
            )

            elapsed = (datetime.now() - start_time).total_seconds()
            total_time += elapsed

            if eval_result["correct"]:
                correct_count += 1
                logger.info(f"  ✅ Correct! ({q['correct_answer']})")
            else:
                logger.info(
                    f"  ❌ Wrong. Expected {q['correct_answer']}, got {eval_result['extracted_answer']}"
                )

            results.append(
                {
                    "id": q["id"],
                    "category": q["category"],
                    "question": q["question"],
                    "correct_answer": q["correct_answer"],
                    "extracted_answer": eval_result["extracted_answer"],
                    "is_correct": eval_result["correct"],
                    "generated_response": generated[:500],
                    "article_reference": q.get("article_reference", ""),
                    "latency_seconds": elapsed,
                }
            )

        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            results.append(
                {
                    "id": q["id"],
                    "category": q["category"],
                    "question": q["question"],
                    "error": str(e),
                    "is_correct": False,
                }
            )

    # Calculate metrics
    accuracy = (correct_count / len(questions)) * 100 if questions else 0
    avg_latency = total_time / len(questions) if questions else 0

    # Category breakdown
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["total"] += 1
        if r.get("is_correct"):
            categories[cat]["correct"] += 1

    summary = {
        "benchmark_name": "Egyptian Law Custom Benchmark",
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(questions),
        "correct_answers": correct_count,
        "accuracy_percent": round(accuracy, 1),
        "avg_latency_seconds": round(avg_latency, 2),
        "total_time_minutes": round(total_time / 60, 1),
        "category_breakdown": {
            cat: {
                "accuracy": round((v["correct"] / v["total"]) * 100, 1),
                "correct": v["correct"],
                "total": v["total"],
            }
            for cat, v in categories.items()
        },
        "results": results,
    }

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to: {output_path}")

    return summary


def print_summary(summary: Dict):
    """Print a formatted summary of benchmark results."""
    print("\n" + "=" * 60)
    print("📊 EGYPTIAN LAW BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\n📈 Overall Performance:")
    print(f"  Questions:       {summary['total_questions']}")
    print(f"  Accuracy:        {summary['accuracy_percent']}%")
    print(f"  Avg Latency:     {summary['avg_latency_seconds']}s")
    print(f"  Total Time:      {summary['total_time_minutes']} minutes")

    print(f"\n📚 Category Breakdown:")
    for cat, metrics in summary["category_breakdown"].items():
        emoji = (
            "✅"
            if metrics["accuracy"] >= 60
            else "⚠️"
            if metrics["accuracy"] >= 40
            else "❌"
        )
        print(
            f"  {emoji} {cat}: {metrics['accuracy']}% ({metrics['correct']}/{metrics['total']})"
        )

    # Grade
    acc = summary["accuracy_percent"]
    if acc >= 80:
        grade = "🏆 EXCELLENT (≥80%)"
    elif acc >= 60:
        grade = "⭐ GOOD (≥60%)"
    elif acc >= 40:
        grade = "⭐ FAIR (≥40%)"
    else:
        grade = "⚠️ NEEDS IMPROVEMENT (<40%)"

    print(f"\n🎯 GRADE: {grade}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run Egyptian Law Custom Benchmark")
    parser.add_argument("--limit", type=int, help="Limit number of questions")
    parser.add_argument(
        "--input",
        type=str,
        default="data/eval/egyptian_law_benchmark.json",
        help="Path to benchmark JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/eval/egyptian_law_results.json",
        help="Path to save results",
    )

    args = parser.parse_args()

    # Load questions
    questions = load_benchmark(args.input)
    logger.info(f"Loaded {len(questions)} questions from {args.input}")

    # Run benchmark
    summary = asyncio.run(
        run_benchmark(questions, limit=args.limit, output_path=args.output)
    )

    # Print summary
    print_summary(summary)

    return 0 if summary["accuracy_percent"] >= 40 else 1


if __name__ == "__main__":
    exit(main())
