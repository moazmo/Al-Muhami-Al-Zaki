"""
Download EgyMMLU Law subset from Hugging Face.

This script downloads the UBC-NLP/EgyMMLU dataset (professional_law config)
to use as a benchmark for Al-Muhami Al-Zaki.

Usage:
    python download_egymmlu.py
"""

from datasets import load_dataset
import json
from pathlib import Path


def main():
    print("=" * 60)
    print("EgyMMLU Dataset Downloader")
    print("=" * 60)

    print("\nLoading EgyMMLU 'professional_law' subset from Hugging Face...")
    print("(This may take a minute on first download)\n")

    # EgyMMLU requires specifying a config (subject)
    # Available configs include: 'professional_law', 'jurisprudence', etc.
    dataset = load_dataset("UBC-NLP/EgyMMLU", "professional_law")

    # Show dataset info
    print("Dataset loaded successfully!")
    print(f"  Test split: {len(dataset['test'])} questions")

    # Use test split
    law_questions = dataset["test"]

    print(f"\n{'=' * 60}")
    print(f"Found {len(law_questions)} professional law questions")
    print(f"{'=' * 60}")

    # Save to file
    output_path = Path("data/eval/egymmlu_law.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    law_data = [dict(q) for q in law_questions]

    # Limit to 50 for reasonable benchmark time
    if len(law_data) > 50:
        print(f"\nLimiting to 50 questions for benchmark (from {len(law_data)})")
        law_data = law_data[:50]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(law_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(law_data)} questions to: {output_path}")

    # Show sample
    print("\n📝 Sample question:")
    sample = law_data[0]
    print(f"  Question: {sample['question']}")
    print(f"  Choices: {sample['choices']}")
    print(f"  Answer: {sample['choices'][sample['answer']]}")
    print(f"  Subject: {sample.get('subject', 'professional_law')}")

    print("\n" + "=" * 60)
    print("Next step: Run the benchmark")
    print("  python scripts/benchmark_egymmlu.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
