"""
Performance Benchmark Script for Al-Muhami Al-Zaki.

Measures latency and throughput across pipeline components:
- Embedding time
- Qdrant search time  
- Grading time (Groq/Llama-3)
- Generation time (Gemini)

Usage:
    python scripts/benchmark.py --iterations 5
"""

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger

from src.utils.logger import setup_logger
from src.utils.config import get_settings


# Test queries for benchmarking
BENCHMARK_QUERIES = [
    "ما هي حقوق الملكية؟",
    "شروط صحة العقد",
    "مدة التقادم",
]


async def benchmark_embedding(iterations: int = 5) -> Dict[str, float]:
    """Benchmark embedding model performance."""
    from src.ingest.embedder import LegalEmbedder
    
    logger.info("Benchmarking embedding...")
    
    # First run is warmup (model loading)
    embedder = LegalEmbedder()
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        embedder.embed_text("ما هي حقوق الملكية في القانون المدني المصري؟", is_query=True)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {
        "component": "embedding",
        "mean_ms": round(statistics.mean(times) * 1000, 2),
        "std_ms": round(statistics.stdev(times) * 1000, 2) if len(times) > 1 else 0,
        "min_ms": round(min(times) * 1000, 2),
        "max_ms": round(max(times) * 1000, 2),
    }


async def benchmark_retrieval(iterations: int = 5) -> Dict[str, float]:
    """Benchmark Qdrant retrieval performance."""
    from src.ingest.embedder import LegalEmbedder
    
    logger.info("Benchmarking retrieval...")
    
    embedder = LegalEmbedder()
    
    times = []
    for query in BENCHMARK_QUERIES[:iterations]:
        start = time.perf_counter()
        embedder.search(query, top_k=5)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {
        "component": "retrieval",
        "mean_ms": round(statistics.mean(times) * 1000, 2),
        "std_ms": round(statistics.stdev(times) * 1000, 2) if len(times) > 1 else 0,
        "min_ms": round(min(times) * 1000, 2),
        "max_ms": round(max(times) * 1000, 2),
    }


async def benchmark_grading(iterations: int = 3) -> Dict[str, float]:
    """Benchmark Groq/Llama-3 grading performance."""
    from langchain_groq import ChatGroq
    from src.prompts.grader import get_grader_prompt
    
    logger.info("Benchmarking grading (Groq/Llama-3)...")
    
    settings = get_settings()
    llm = ChatGroq(
        model=settings.grader_model,
        api_key=settings.groq_api_key,
        temperature=0.0,
    )
    
    test_doc = "المادة 147 - العقد شريعة المتعاقدين"
    test_question = "ما هي قوة العقد القانونية؟"
    
    times = []
    for _ in range(iterations):
        prompt = get_grader_prompt(test_question, test_doc)
        start = time.perf_counter()
        await llm.ainvoke(prompt)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {
        "component": "grading_groq",
        "mean_ms": round(statistics.mean(times) * 1000, 2),
        "std_ms": round(statistics.stdev(times) * 1000, 2) if len(times) > 1 else 0,
        "min_ms": round(min(times) * 1000, 2),
        "max_ms": round(max(times) * 1000, 2),
    }


async def benchmark_generation(iterations: int = 3) -> Dict[str, float]:
    """Benchmark Gemini generation performance."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from src.prompts.generator import get_generator_prompt
    
    logger.info("Benchmarking generation (Gemini)...")
    
    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model=settings.generator_model,
        google_api_key=settings.google_api_key,
        temperature=0.3,
    )
    
    test_context = "المادة 147 - العقد شريعة المتعاقدين، فلا يجوز نقضه أو تعديله إلا باتفاق الطرفين."
    test_question = "ما هي قوة العقد القانونية؟"
    
    times = []
    for _ in range(iterations):
        prompt = get_generator_prompt(test_question, test_context)
        start = time.perf_counter()
        await llm.ainvoke(prompt)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {
        "component": "generation_gemini",
        "mean_ms": round(statistics.mean(times) * 1000, 2),
        "std_ms": round(statistics.stdev(times) * 1000, 2) if len(times) > 1 else 0,
        "min_ms": round(min(times) * 1000, 2),
        "max_ms": round(max(times) * 1000, 2),
    }


async def benchmark_full_pipeline(iterations: int = 3) -> Dict[str, float]:
    """Benchmark full CRAG pipeline end-to-end."""
    from src.graph.builder import run_query
    
    logger.info("Benchmarking full CRAG pipeline...")
    
    times = []
    for i, query in enumerate(BENCHMARK_QUERIES[:iterations]):
        logger.info(f"  Pipeline run {i+1}/{iterations}")
        start = time.perf_counter()
        await run_query(query)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {
        "component": "full_pipeline",
        "mean_ms": round(statistics.mean(times) * 1000, 2),
        "std_ms": round(statistics.stdev(times) * 1000, 2) if len(times) > 1 else 0,
        "min_ms": round(min(times) * 1000, 2),
        "max_ms": round(max(times) * 1000, 2),
    }


async def run_benchmark(iterations: int = 5) -> List[Dict[str, Any]]:
    """Run all benchmarks."""
    logger.info("=" * 60)
    logger.info("Al-Muhami Al-Zaki — Performance Benchmark")
    logger.info("=" * 60)
    
    results = []
    
    # Run each benchmark
    results.append(await benchmark_embedding(iterations))
    results.append(await benchmark_retrieval(iterations))
    results.append(await benchmark_grading(min(iterations, 3)))  # Limit API calls
    results.append(await benchmark_generation(min(iterations, 3)))
    results.append(await benchmark_full_pipeline(min(iterations, 3)))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Component':<20} {'Mean (ms)':<12} {'Std (ms)':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)
    
    for r in results:
        print(
            f"{r['component']:<20} "
            f"{r['mean_ms']:<12} "
            f"{r['std_ms']:<10} "
            f"{r['min_ms']:<10} "
            f"{r['max_ms']:<10}"
        )
    
    print("=" * 60)
    
    # Performance grades
    full_pipeline_mean = next(r for r in results if r["component"] == "full_pipeline")["mean_ms"]
    
    print("\n🎯 PERFORMANCE GRADE:")
    if full_pipeline_mean < 5000:
        print("  ⭐⭐⭐ EXCELLENT (<5s)")
    elif full_pipeline_mean < 15000:
        print("  ⭐⭐ GOOD (5-15s)")
    elif full_pipeline_mean < 30000:
        print("  ⭐ ACCEPTABLE (15-30s)")
    else:
        print("  ⚠️ NEEDS OPTIMIZATION (>30s)")
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run performance benchmark on Al-Muhami Al-Zaki"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations per benchmark (default: 5)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logger(level="INFO")
    
    try:
        asyncio.run(run_benchmark(args.iterations))
        return 0
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
