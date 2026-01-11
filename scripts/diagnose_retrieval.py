#!/usr/bin/env python3
"""Diagnose retrieval issues for benchmark questions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.embedder import LegalEmbedder

# Test questions from benchmark
test_questions = [
    "ما هي عقوبة السرقة البسيطة في قانون العقوبات المصري؟",
    "ما هي عقوبة القتل العمد مع سبق الإصرار والترصد؟",
    "بموجب القانون المدني المصري، ما هو الأثر القانوني لعدم الوفاء بالالتزام التعاقدي؟",
    "ما هي مدة ولاية رئيس الجمهورية وفقاً للدستور المصري؟",
    "ما هو الحد الأدنى لسن الزواج للفتاة في قانون الأحوال الشخصية المصري؟",
]

print("Initializing embedder...")
embedder = LegalEmbedder()

for q in test_questions:
    print(f"\n{'=' * 60}")
    print(f"QUERY: {q[:60]}...")
    print("=" * 60)

    results = embedder.search(q, top_k=3)

    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        source = r.get("source_name", "Unknown")
        article = r.get("article_number", "N/A")
        text = r.get("text", "")[:150]

        print(f"\n[{i}] Score: {score:.3f} | Source: {source} | Article: {article}")
        print(f"    Text: {text}...")
