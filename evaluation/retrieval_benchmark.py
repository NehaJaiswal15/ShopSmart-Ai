"""
Retrieval Strategy Benchmark

Compares vector-only, BM25-only, and hybrid retrieval strategies
by running the full RAG evaluation pipeline with each one.

This produces a side-by-side comparison showing which strategy
performs best on each category of queries.

Usage:
    python -m evaluation.retrieval_benchmark

Note: Takes ~10-15 minutes (runs evaluation 3 times — once per strategy).
"""
import os
import sys
import json
import time
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flipkart.data_ingestion import DataIngestor
from flipkart.rag_chain import RAGChainBuilder
from evaluation.rag_evaluator import load_test_set, run_evaluation


STRATEGIES = ["vector", "bm25", "hybrid"]
RESULTS_PATH = "evaluation/results"


def main():
    print("=" * 70)
    print("Retrieval Strategy Benchmark")
    print("=" * 70)
    
    # Load test set
    test_cases = load_test_set()
    print(f"\nTest set: {len(test_cases)} cases")
    
    # Initialize vector store (shared across strategies)
    print("\nInitializing AstraDB connection...")
    vector_store = DataIngestor().ingest(load_existing=True)
    
    all_results = {}
    
    for strategy in STRATEGIES:
        print(f"\n{'=' * 70}")
        print(f"STRATEGY: {strategy.upper()}")
        print(f"{'=' * 70}")
        
        # Build chain with this strategy
        rag_chain = RAGChainBuilder(vector_store, retrieval_strategy=strategy).build_chain()
        
        # Run evaluation
        start = time.time()
        evaluation = run_evaluation(rag_chain, test_cases)
        elapsed = time.time() - start
        
        evaluation["aggregate"]["strategy"] = strategy
        evaluation["aggregate"]["total_time_seconds"] = round(elapsed, 1)
        all_results[strategy] = evaluation
    
    # Print comparison
    print("\n\n" + "=" * 70)
    print("COMPARISON: All Strategies")
    print("=" * 70)
    
    print(f"\n  {'Metric':<25} {'Vector':>10} {'BM25':>10} {'Hybrid':>10}")
    print(f"  {'─' * 55}")
    
    for metric in ["answer_correctness", "context_relevance", "faithfulness"]:
        scores = []
        for strategy in STRATEGIES:
            score = all_results[strategy]["aggregate"]["avg_scores"].get(metric, 0)
            scores.append(score)
        
        best_idx = scores.index(max(scores))
        row = f"  {metric:<25}"
        for i, score in enumerate(scores):
            marker = " ★" if i == best_idx else "  "
            row += f" {score:>8.3f}{marker}"
        print(row)
    
    # Latency
    latencies = [all_results[s]["aggregate"]["avg_latency_seconds"] for s in STRATEGIES]
    print(f"\n  {'Avg Latency (s)':<25}", end="")
    for l in latencies:
        print(f" {l:>10.2f}", end="")
    print()
    
    # Per-category comparison
    print(f"\n  Context Relevance by Category:")
    print(f"  {'Category':<20} {'Vector':>10} {'BM25':>10} {'Hybrid':>10}")
    print(f"  {'─' * 50}")
    
    all_cats = set()
    for s in STRATEGIES:
        all_cats.update(all_results[s]["aggregate"]["scores_by_category"].keys())
    
    for cat in sorted(all_cats):
        row = f"  {cat:<20}"
        for strategy in STRATEGIES:
            cat_data = all_results[strategy]["aggregate"]["scores_by_category"].get(cat, {})
            score = cat_data.get("context_relevance", 0)
            row += f" {score:>10.2f}"
        print(row)
    
    # Save combined results
    os.makedirs(RESULTS_PATH, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(RESULTS_PATH, f"benchmark_{timestamp}.json")
    
    benchmark = {
        "timestamp": timestamp,
        "strategies": STRATEGIES,
        "comparison": {
            s: all_results[s]["aggregate"] for s in STRATEGIES
        },
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Results saved to: {filepath}")
    
    # Recommendation
    best_strategy = max(STRATEGIES, key=lambda s: sum(all_results[s]["aggregate"]["avg_scores"].values()))
    print(f"\n  ✅ Recommended strategy: {best_strategy.upper()}")
    print(f"     Overall score: {sum(all_results[best_strategy]['aggregate']['avg_scores'].values()) / 3:.3f}")


if __name__ == "__main__":
    main()
