"""
Hallucination check via self-consistency evaluation.

This script:
- Generates multiple outputs (10) for the same question with higher temperature
- Compares them semantically to detect hallucinations/inconsistency
- Reports average pairwise similarity and consistency scores

Technique: Self-consistency checking / semantic consistency evaluation
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "qwen3-0.6b-finetuned"

# Test questions (you can replace with your own)
TEST_QUESTIONS = [
    "What is GraphRAG?",
    "How should I handle my dreams about a person?",
    "What is Neo4j?",
]


def generate_multiple_answers(
    model_pipeline,
    question: str,
    num_samples: int = 10,
    temperature: float = 0.8,
) -> List[str]:
    """Generate multiple answers to the same question with higher temperature.

    Parameters
    ----------
    model_pipeline:
        The text-generation pipeline.
    question:
        The question to ask.
    num_samples:
        How many outputs to generate (default: 10).
    temperature:
        Sampling temperature (higher = more diverse outputs).

    Returns
    -------
    List[str]
        List of generated answers.
    """

    prompt = f"Question: {question}\nAnswer:"
    answers: List[str] = []

    for _ in range(num_samples):
        output = model_pipeline(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
        )[0]["generated_text"]

        # Strip the prompt part
        answer = output[len(prompt) :].strip()
        answers.append(answer)

    return answers


def compute_pairwise_similarities(answers: List[str], embed_model) -> np.ndarray:
    """Compute pairwise cosine similarities between all answers.

    Returns
    -------
    np.ndarray
        Symmetric matrix of shape (len(answers), len(answers)) with
        pairwise cosine similarities. Diagonal is 1.0 (self-similarity).
    """

    if len(answers) < 2:
        return np.array([[1.0]])

    # Embed all answers
    embeddings = embed_model.encode(answers, convert_to_tensor=True)

    # Compute pairwise cosine similarities
    similarity_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()

    return similarity_matrix


def analyze_consistency(
    answers: List[str],
    similarity_matrix: np.ndarray,
) -> dict:
    """Analyze consistency metrics from pairwise similarities.

    Returns
    -------
    dict
        Metrics including:
        - avg_pairwise_similarity: average of all pairwise similarities (excluding diagonal)
        - min_similarity: minimum pairwise similarity
        - max_similarity: maximum pairwise similarity
        - std_similarity: standard deviation of pairwise similarities
        - consistency_score: heuristic score (0-1, higher = more consistent)
    """

    # Get upper triangle (excluding diagonal) to avoid double-counting
    n = len(answers)
    if n < 2:
        return {
            "avg_pairwise_similarity": 1.0,
            "min_similarity": 1.0,
            "max_similarity": 1.0,
            "std_similarity": 0.0,
            "consistency_score": 1.0,
        }

    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
    pairwise_sims = similarity_matrix[mask]

    avg_sim = float(np.mean(pairwise_sims))
    min_sim = float(np.min(pairwise_sims))
    max_sim = float(np.max(pairwise_sims))
    std_sim = float(np.std(pairwise_sims))

    # Heuristic consistency score:
    # - High average similarity (>= 0.85) = consistent
    # - Low std = consistent
    # - Scale to 0-1 range
    consistency_score = min(1.0, avg_sim * (1.0 - min(std_sim, 0.2)))

    return {
        "avg_pairwise_similarity": avg_sim,
        "min_similarity": min_sim,
        "max_similarity": max_sim,
        "std_similarity": std_sim,
        "consistency_score": consistency_score,
    }


def main() -> None:
    """Run hallucination check via self-consistency evaluation."""

    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )

    print("Loading sentence embedding model for semantic comparison...")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("\n" + "=" * 80)
    print("HALLUCINATION CHECK: Self-Consistency Evaluation")
    print("=" * 80)
    print(
        "\nGenerating 10 outputs per question with temperature=0.8, "
        "then comparing semantically.\n"
    )

    all_consistency_scores: List[float] = []

    for idx, question in enumerate(TEST_QUESTIONS, start=1):
        print(f"\n[{idx}/{len(TEST_QUESTIONS)}] Question: {question!r}")
        print("-" * 80)

        # Generate 10 answers
        answers = generate_multiple_answers(
            gen_pipeline,
            question,
            num_samples=10,
            temperature=0.8,
        )

        # Compute pairwise similarities
        similarity_matrix = compute_pairwise_similarities(answers, embed_model)

        # Analyze consistency
        metrics = analyze_consistency(answers, similarity_matrix)

        # Print results
        print(f"\nGenerated {len(answers)} answers:")
        for i, ans in enumerate(answers, start=1):
            print(f"  {i}. {ans[:100]}{'...' if len(ans) > 100 else ''}")

        print(f"\nConsistency Metrics:")
        print(
            f"  Average pairwise similarity: {metrics['avg_pairwise_similarity']:.4f}"
        )
        print(f"  Min similarity: {metrics['min_similarity']:.4f}")
        print(f"  Max similarity: {metrics['max_similarity']:.4f}")
        print(f"  Std deviation: {metrics['std_similarity']:.4f}")
        print(f"  Consistency score: {metrics['consistency_score']:.4f}")

        # Interpretation
        if metrics["consistency_score"] >= 0.85:
            print("  → HIGH CONSISTENCY (likely grounded, low hallucination)")
        elif metrics["consistency_score"] >= 0.70:
            print("  → MODERATE CONSISTENCY (some variance, possible hallucinations)")
        else:
            print("  → LOW CONSISTENCY (high variance, likely hallucinations)")

        all_consistency_scores.append(metrics["consistency_score"])

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    overall_avg = sum(all_consistency_scores) / len(all_consistency_scores)
    print(f"\nAverage consistency score across all questions: {overall_avg:.4f}")

    if overall_avg >= 0.85:
        print("→ Model shows HIGH overall consistency (low hallucination risk)")
    elif overall_avg >= 0.70:
        print("→ Model shows MODERATE overall consistency")
    else:
        print("→ Model shows LOW overall consistency (high hallucination risk)")


if __name__ == "__main__":
    main()
