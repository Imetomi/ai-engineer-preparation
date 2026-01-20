"""
Evaluate the fine-tuned Qwen3-0.6B-Base model with ROUGE-L on the tiny dataset.

This:
- loads `model-training/tiny_dataset.jsonl`
- parses (question, reference_answer) from each line
- generates an answer with the fine-tuned model
- computes ROUGE-L between generated answers and references
- prints the final ROUGE-L score
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import evaluate
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "qwen3-0.6b-finetuned"
DATA_PATH = ROOT / "tiny_dataset.jsonl"


def parse_qa(text: str) -> Tuple[str, str]:
    """Parse 'Question: ...\\nAnswer: ...' -> (question, answer)."""

    # Very simple parser tailored to how tiny_dataset.jsonl is formatted.
    # If the format changes, adjust this accordingly.
    q_prefix = "Question:"
    a_prefix = "\nAnswer:"

    if q_prefix not in text or a_prefix not in text:
        raise ValueError(f"Unexpected text format: {text!r}")

    q_part, a_part = text.split(a_prefix, 1)
    question = q_part[len(q_prefix) :].strip()
    answer = a_part.strip()
    return question, answer


def load_eval_cases() -> List[Tuple[str, str]]:
    """Load (question, reference_answer) pairs from the tiny dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_PATH}")

    dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")
    cases: List[Tuple[str, str]] = []
    for row in dataset:
        q, a = parse_qa(row["text"])
        cases.append((q, a))
    return cases


def build_generator():
    """Load the fine-tuned model and build a generation pipeline."""

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Fine-tuned model directory not found: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True)

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )
    return gen


def generate_answer(gen, question: str) -> str:
    """Generate an answer for a given question using the fine-tuned model."""

    prompt = f"Question: {question}\nAnswer:"
    out = gen(
        prompt,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
    )[
        0
    ]["generated_text"]
    generated = out[len(prompt) :].strip()
    return generated


def main() -> None:
    print("Loading eval cases...")
    cases = load_eval_cases()
    print(f"Loaded {len(cases)} examples from {DATA_PATH}")

    print("Loading fine-tuned model for generation...")
    gen = build_generator()

    predictions: List[str] = []
    references: List[str] = []

    for i, (question, reference) in enumerate(cases, start=1):
        print(f"\n[{i}/{len(cases)}] Question: {question}")
        print(f"Reference answer: {reference}")

        pred = generate_answer(gen, question)
        print(f"Generated answer: {pred}")

        predictions.append(pred)
        references.append(reference)

    print("\nComputing ROUGE-L...")
    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=predictions, references=references)

    rouge_l = scores.get("rougeL")
    print(f"\nFinal ROUGE-L score: {rouge_l:.4f}")


if __name__ == "__main__":
    main()
