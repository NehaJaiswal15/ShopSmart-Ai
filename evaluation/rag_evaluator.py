"""
RAG Evaluation Pipeline

Evaluates the quality of the RAG chatbot using a curated golden test set.
Measures both retrieval quality and answer quality.

Metrics:
    - Answer Correctness: Does the response contain expected keywords?
    - Context Relevance: Are retrieved docs about the right product?
    - Faithfulness: Is the answer grounded in retrieved context (not hallucinated)?
    - Response Coverage: What % of test cases get acceptable answers?

Usage:
    python -m evaluation.rag_evaluator
"""
import os
import sys
import json
import time
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flipkart.data_ingestion import DataIngestor
from flipkart.rag_chain import RAGChainBuilder
from utils.logger import get_logger

logger = get_logger(__name__)

TEST_SET_PATH = "evaluation/test_set.json"
RESULTS_PATH = "evaluation/results"


def load_test_set(path: str = TEST_SET_PATH) -> list:
    """Load the golden test set."""
    with open(path, "r") as f:
        data = json.load(f)
    return data["test_cases"]


def evaluate_answer_correctness(answer: str, expected_contains: list) -> dict:
    """Check if the answer contains expected keywords/phrases.
    
    This is a simple but effective check — if we ask about BoAt Rockerz price
    and the answer contains '999', it's likely correct.
    
    Returns:
        dict with score (0-1), matched keywords, and missed keywords.
    """
    if not expected_contains:
        return {"score": 1.0, "matched": [], "missed": [], "note": "No expected content specified"}
    
    answer_lower = answer.lower()
    matched = [kw for kw in expected_contains if kw.lower() in answer_lower]
    missed = [kw for kw in expected_contains if kw.lower() not in answer_lower]
    
    score = len(matched) / len(expected_contains) if expected_contains else 1.0
    return {"score": round(score, 2), "matched": matched, "missed": missed}


def evaluate_context_relevance(docs: list, expected_products: list) -> dict:
    """Check if retrieved documents are about the expected product(s).
    
    Measures whether the retriever found the right product reviews.
    
    Returns:
        dict with score (0-1) and details.
    """
    if not expected_products or not docs:
        return {"score": 1.0 if not expected_products else 0.0, "retrieved_products": [], "expected": expected_products}
    
    # Extract product names from retrieved doc metadata
    retrieved_texts = [doc.page_content.lower() for doc in docs]
    
    hits = 0
    for product in expected_products:
        product_lower = product.lower()
        # Check if any retrieved doc mentions this product
        if any(product_lower in text or product_lower[:20] in text for text in retrieved_texts):
            hits += 1
    
    score = hits / len(expected_products) if expected_products else 1.0
    return {
        "score": round(score, 2),
        "expected": expected_products,
        "docs_retrieved": len(docs),
    }


def evaluate_faithfulness(answer: str, docs: list) -> dict:
    """Check if the answer is grounded in retrieved context.
    
    A simple faithfulness check: extracts key claims from the answer
    and verifies they appear in the retrieved documents.
    
    This is a lightweight alternative to using an LLM-based judge.
    
    Returns:
        dict with score (0-1) and details.
    """
    if not docs:
        return {"score": 0.5, "note": "No documents retrieved"}
    
    context = " ".join([doc.page_content.lower() for doc in docs])
    answer_lower = answer.lower()
    
    # Extract potential factual claims (numbers, product names)
    import re
    numbers_in_answer = re.findall(r'\b\d{3,4}\b', answer)  # 3-4 digit numbers (prices)
    
    if not numbers_in_answer:
        return {"score": 0.8, "note": "No numerical claims to verify"}
    
    # Check if numbers in the answer exist in context
    grounded = [n for n in numbers_in_answer if n in context]
    ungrounded = [n for n in numbers_in_answer if n not in context]
    
    score = len(grounded) / len(numbers_in_answer) if numbers_in_answer else 1.0
    return {
        "score": round(score, 2),
        "grounded_claims": grounded,
        "ungrounded_claims": ungrounded,
    }


def run_evaluation(rag_chain, test_cases: list) -> dict:
    """Run full evaluation against the golden test set.
    
    Args:
        rag_chain: The configured RAG chain to evaluate.
        test_cases: List of test case dicts from the golden test set.
    
    Returns:
        dict with per-case results and aggregate metrics.
    """
    results = []
    total_time = 0
    
    print(f"\nEvaluating {len(test_cases)} test cases...")
    print("=" * 70)
    
    for i, tc in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] {tc['category']}: {tc['question'][:50]}...")
        
        start = time.time()
        try:
            result = rag_chain.invoke(
                {"input": tc["question"]},
                config={"configurable": {"session_id": f"eval-{tc['id']}"}}
            )
            elapsed = time.time() - start
            total_time += elapsed
            
            answer = result["answer"]
            docs = result.get("context", [])
            
            # Run evaluations
            correctness = evaluate_answer_correctness(answer, tc["expected_answer_contains"])
            relevance = evaluate_context_relevance(docs, tc["expected_products"])
            faithfulness = evaluate_faithfulness(answer, docs)
            
            case_result = {
                "id": tc["id"],
                "category": tc["category"],
                "question": tc["question"],
                "answer": answer[:200] + "..." if len(answer) > 200 else answer,
                "latency_seconds": round(elapsed, 2),
                "scores": {
                    "answer_correctness": correctness["score"],
                    "context_relevance": relevance["score"],
                    "faithfulness": faithfulness["score"],
                },
                "details": {
                    "correctness": correctness,
                    "relevance": relevance,
                    "faithfulness": faithfulness,
                }
            }
            
            avg_score = sum(case_result["scores"].values()) / len(case_result["scores"])
            status = "✅" if avg_score >= 0.6 else "⚠️" if avg_score >= 0.3 else "❌"
            print(f"  {status} Correctness={correctness['score']:.1f} | Relevance={relevance['score']:.1f} | Faithfulness={faithfulness['score']:.1f} | {elapsed:.1f}s")
            
        except Exception as e:
            elapsed = time.time() - start
            case_result = {
                "id": tc["id"],
                "category": tc["category"],
                "question": tc["question"],
                "error": str(e),
                "latency_seconds": round(elapsed, 2),
                "scores": {"answer_correctness": 0, "context_relevance": 0, "faithfulness": 0},
            }
            print(f"  ❌ Error: {e}")
        
        results.append(case_result)
    
    # Aggregate metrics
    valid_results = [r for r in results if "error" not in r]
    
    aggregate = {
        "total_cases": len(test_cases),
        "successful_cases": len(valid_results),
        "failed_cases": len(test_cases) - len(valid_results),
        "avg_latency_seconds": round(total_time / len(test_cases), 2),
        "avg_scores": {},
        "scores_by_category": {},
    }
    
    if valid_results:
        for metric in ["answer_correctness", "context_relevance", "faithfulness"]:
            scores = [r["scores"][metric] for r in valid_results]
            aggregate["avg_scores"][metric] = round(sum(scores) / len(scores), 3)
        
        # Group by category
        categories = set(r["category"] for r in valid_results)
        for cat in sorted(categories):
            cat_results = [r for r in valid_results if r["category"] == cat]
            cat_scores = {}
            for metric in ["answer_correctness", "context_relevance", "faithfulness"]:
                s = [r["scores"][metric] for r in cat_results]
                cat_scores[metric] = round(sum(s) / len(s), 3)
            cat_scores["count"] = len(cat_results)
            aggregate["scores_by_category"][cat] = cat_scores
    
    return {"aggregate": aggregate, "results": results}


def print_report(evaluation: dict):
    """Print a formatted evaluation report."""
    agg = evaluation["aggregate"]
    
    print("\n" + "=" * 70)
    print("RAG EVALUATION REPORT")
    print("=" * 70)
    
    print(f"\n  Total test cases:    {agg['total_cases']}")
    print(f"  Successful:          {agg['successful_cases']}")
    print(f"  Failed:              {agg['failed_cases']}")
    print(f"  Avg latency:         {agg['avg_latency_seconds']}s")
    
    print(f"\n  Overall Scores:")
    for metric, score in agg["avg_scores"].items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"    {metric:<25} {bar} {score:.3f}")
    
    print(f"\n  Scores by Category:")
    print(f"    {'Category':<20} {'Correct':>8} {'Relevant':>9} {'Faithful':>9} {'Count':>6}")
    print(f"    {'─' * 52}")
    for cat, scores in sorted(agg["scores_by_category"].items()):
        print(f"    {cat:<20} {scores['answer_correctness']:>8.2f} {scores['context_relevance']:>9.2f} {scores['faithfulness']:>9.2f} {scores['count']:>6}")
    
    # Flag worst performing cases
    print(f"\n  Lowest Scoring Cases:")
    results_with_avg = [
        (r, sum(r["scores"].values()) / 3)
        for r in evaluation["results"]
        if "error" not in r
    ]
    results_with_avg.sort(key=lambda x: x[1])
    for r, avg in results_with_avg[:5]:
        print(f"    ⚠️  [{r['id']}] avg={avg:.2f} — {r['question'][:50]}")


def save_results(evaluation: dict, path: str = RESULTS_PATH):
    """Save evaluation results to JSON."""
    os.makedirs(path, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(path, f"eval_{timestamp}.json")
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Results saved to: {filepath}")
    return filepath


def main():
    print("=" * 70)
    print("RAG Evaluation Pipeline")
    print("=" * 70)
    
    # Load test set
    print("\nLoading golden test set...")
    test_cases = load_test_set()
    print(f"  {len(test_cases)} test cases loaded")
    categories = {}
    for tc in test_cases:
        categories[tc["category"]] = categories.get(tc["category"], 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"    {cat}: {count}")
    
    # Initialize RAG chain
    print("\nInitializing RAG pipeline...")
    vector_store = DataIngestor().ingest(load_existing=True)
    rag_chain = RAGChainBuilder(vector_store).build_chain()
    print("  Ready!")
    
    # Run evaluation
    evaluation = run_evaluation(rag_chain, test_cases)
    
    # Report
    print_report(evaluation)
    
    # Save
    filepath = save_results(evaluation)
    
    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
