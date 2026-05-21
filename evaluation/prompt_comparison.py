"""
Prompt Engineering Comparison

Tests multiple system prompts against the golden test set
to systematically find which prompt produces the best answers.

Why this matters:
- Prompt engineering is often done by "feel" — this makes it data-driven
- Small prompt changes can have big impacts on faithfulness and correctness
- Shows interviewers you treat prompt design as an engineering process

Usage:
    python -m evaluation.prompt_comparison

Note: Takes ~15-20 minutes (runs evaluation once per prompt variant).
"""

import os
import sys
import json
import time
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain  # type: ignore
from langchain_classic.chains.combine_documents import create_stuff_documents_chain  # type: ignore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from flipkart.config import Config
from flipkart.data_ingestion import DataIngestor
from flipkart.retrieval import get_retriever
from evaluation.rag_evaluator import load_test_set, run_evaluation

RESULTS_PATH = "evaluation/results"

# ── Prompt Variants ────────────────────────────────────────────

PROMPTS = {
    "minimal": {
        "description": "Bare minimum — just says to use context",
        "template": """Answer the user's question based on the context below.

CONTEXT:
{context}

QUESTION: {input}""",
    },
    "current": {
        "description": "Our current production prompt with anti-hallucination rules",
        "template": """You are ShopSmart AI, an e-commerce assistant that helps users with product-related queries.

You have access to product reviews, ratings, and prices. Use ONLY the information provided in the context below.

Rules:
- Always mention the actual price when discussing a product (prices are in ₹ INR).
- Use ratings and review summaries to give helpful recommendations.
- If the user asks about something NOT in the context, say "I don't have information about that product."
- Never make up prices, ratings, or product details.
- Be concise and helpful.

CONTEXT:
{context}

QUESTION: {input}""",
    },
    "structured": {
        "description": "Forces structured output with clear sections",
        "template": """You are ShopSmart AI, a product recommendation assistant.

INSTRUCTIONS:
1. Answer ONLY using the product information provided in CONTEXT below.
2. When discussing a product, always include: Product Name, Price (₹), Rating (/5).
3. If comparing products, use a clear format with pros and cons for each.
4. If the question is about a product NOT in the context, respond: "I don't have information about that product in my database."
5. NEVER invent or guess prices, ratings, or features.
6. Keep responses concise — 2-3 paragraphs maximum.

CONTEXT:
{context}

QUESTION: {input}""",
    },
    "expert": {
        "description": "Persona-driven prompt with expertise framing",
        "template": """You are ShopSmart AI, an expert audio product advisor with deep knowledge of headphones, earbuds, and neckbands.

Your role is to help customers make informed purchase decisions based on real user reviews and verified product data.

IMPORTANT RULES:
- Base ALL recommendations on the CONTEXT provided. Never fabricate information.
- Always quote exact prices in ₹ when available.
- Highlight both positives AND negatives from reviews for balanced advice.
- When comparing products, consider: sound quality, build quality, battery life, comfort, and value for money.
- If a product is not in your database, say so honestly.
- Format prices as ₹XXX (e.g., ₹999, ₹1,499).

CONTEXT:
{context}

QUESTION: {input}""",
    },
}


# ── Build Chain with Custom Prompt ─────────────────────────────


def build_chain_with_prompt(vector_store, prompt_template: str):
    """Build a RAG chain using a specific system prompt."""
    model = ChatGroq(model=Config.RAG_MODEL, temperature=0.5)
    retriever = get_retriever(vector_store, strategy="hybrid", k=3)
    history_store = {}

    def get_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in history_store:
            history_store[session_id] = ChatMessageHistory()
        return history_store[session_id]

    context_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Given the chat history and user question, rewrite it as a standalone question."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", prompt_template), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")]
    )

    history_aware_retriever = create_history_aware_retriever(model, retriever, context_prompt)
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


# ── Main ───────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("Prompt Engineering Comparison")
    print("=" * 70)

    test_cases = load_test_set()
    print(f"\nTest set: {len(test_cases)} cases")

    print("\nInitializing RAG pipeline...")
    vector_store = DataIngestor().ingest(load_existing=True)

    all_results = {}

    for name, prompt_info in PROMPTS.items():
        print(f"\n{'=' * 70}")
        print(f"PROMPT: {name.upper()}")
        print(f"Description: {prompt_info['description']}")
        print(f"{'=' * 70}")

        chain = build_chain_with_prompt(vector_store, prompt_info["template"])

        start = time.time()
        evaluation = run_evaluation(chain, test_cases)
        elapsed = time.time() - start

        evaluation["aggregate"]["prompt"] = name
        evaluation["aggregate"]["prompt_description"] = prompt_info["description"]
        evaluation["aggregate"]["total_time_seconds"] = round(elapsed, 1)
        all_results[name] = evaluation

    # Print comparison
    print("\n\n" + "=" * 70)
    print("PROMPT COMPARISON RESULTS")
    print("=" * 70)

    print(f"\n  {'Prompt':<15} {'Correctness':>12} {'Relevance':>10} {'Faithful':>9} {'Avg':>8} {'Latency':>9}")
    print(f"  {'─' * 63}")

    best_prompt = None
    best_avg = 0

    for name in PROMPTS:
        scores = all_results[name]["aggregate"]["avg_scores"]
        latency = all_results[name]["aggregate"]["avg_latency_seconds"]
        avg = sum(scores.values()) / len(scores)
        if avg > best_avg:
            best_avg = avg
            best_prompt = name

        print(
            f"  {name:<15} {scores['answer_correctness']:>12.3f} {scores['context_relevance']:>10.3f} {scores['faithfulness']:>9.3f} {avg:>8.3f} {latency:>8.1f}s"
        )

    # Mark the winner
    print(f"\n  ✅ Best prompt: {best_prompt.upper()} (avg score: {best_avg:.3f})")

    # Per-category comparison for top 2 prompts
    sorted_prompts = sorted(
        PROMPTS.keys(), key=lambda p: sum(all_results[p]["aggregate"]["avg_scores"].values()), reverse=True
    )
    top2 = sorted_prompts[:2]

    print(f"\n  Category Breakdown ({top2[0]} vs {top2[1]}):")
    print(f"  {'Category':<20} {top2[0]:>12} {top2[1]:>12}")
    print(f"  {'─' * 44}")

    all_cats = set()
    for p in top2:
        all_cats.update(all_results[p]["aggregate"]["scores_by_category"].keys())

    for cat in sorted(all_cats):
        row = f"  {cat:<20}"
        for p in top2:
            cat_data = all_results[p]["aggregate"]["scores_by_category"].get(cat, {})
            avg = sum(v for k, v in cat_data.items() if k != "count") / 3
            row += f" {avg:>12.2f}"
        print(row)

    # Save results
    os.makedirs(RESULTS_PATH, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(RESULTS_PATH, f"prompts_{timestamp}.json")

    summary = {
        "timestamp": timestamp,
        "prompts_tested": list(PROMPTS.keys()),
        "winner": best_prompt,
        "comparison": {
            name: {
                "description": PROMPTS[name]["description"],
                "scores": all_results[name]["aggregate"]["avg_scores"],
                "avg_score": round(sum(all_results[name]["aggregate"]["avg_scores"].values()) / 3, 3),
                "latency": all_results[name]["aggregate"]["avg_latency_seconds"],
                "by_category": all_results[name]["aggregate"]["scores_by_category"],
            }
            for name in PROMPTS
        },
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {filepath}")
    print(f"\n{'=' * 70}")
    print("Done! Update rag_chain.py with the winning prompt.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
