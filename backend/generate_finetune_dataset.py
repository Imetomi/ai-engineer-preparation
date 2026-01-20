"""
Helper script to generate a fine-tuning dataset from your GraphRAG queries.

This collects question-answer pairs that you can manually review/edit,
then use for fine-tuning your model.

Usage:
    python -m backend.generate_finetune_dataset
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from agent import query


def generate_dataset_from_queries(
    questions: List[str],
    output_path: str | Path = "finetune_dataset.json",
) -> None:
    """Generate a fine-tuning dataset by running queries and collecting answers.

    Parameters
    ----------
    questions:
        List of questions to ask your GraphRAG agent.
    output_path:
        Where to save the JSON dataset file.

    Notes
    -----
    - This will call your agent.query() for each question.
    - You should MANUALLY REVIEW and EDIT the answers before using for fine-tuning.
    - The dataset format matches what Unsloth/transformers expect.
    """

    dataset: List[dict] = []

    print(f"Generating dataset from {len(questions)} questions...\n")

    for idx, question in enumerate(questions, start=1):
        print(f"[{idx}/{len(questions)}] Querying: {question!r}")

        try:
            # Call your agent to get an answer
            answer = query(question)

            # Format as instruction-following example
            example = {
                "instruction": "Answer based on the provided context from my personal notes. Be concise, structured, and clear. Always reply in English.",
                "input": f"Question: {question}",
                "output": answer,
            }

            dataset.append(example)
            print(f"  ✓ Got answer ({len(answer)} chars)\n")

        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            continue

    # Save to JSON
    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(dataset)} examples to: {output_path}")
    print("\n⚠️  IMPORTANT: Review and edit the answers before using for fine-tuning!")
    print("   - Fix any hallucinations")
    print("   - Ensure answers match your desired style")
    print("   - Remove any examples that aren't good quality")


if __name__ == "__main__":
    # Example questions - replace with your actual questions
    example_questions = [
        "How should I handle my dreams about Rebi?",
        "What is GraphRAG and how does it work?",
        "Explain how Neo4j stores my notes.",
        "What are the main concepts in my notes about AI?",
        # Add more questions here...
    ]

    generate_dataset_from_queries(
        questions=example_questions,
        output_path="finetune_dataset.json",
    )
