"""
Minimal fine-tuning script for Qwen3-0.6B-Base on a tiny custom dataset.

Goal: show you the **actual code** to:
- load the model from Hugging Face
- load a tiny JSONL dataset
- fine-tune for a couple of epochs
- save the resulting model

This is NOT about getting a great model, just about understanding and
running the full training loop end-to-end.
"""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
DATA_PATH = Path(__file__).parent / "tiny_dataset.jsonl"
OUTPUT_DIR = Path(__file__).parent / "qwen3-0.6b-finetuned"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_PATH}")

    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    print(f"Loading dataset from: {DATA_PATH}")
    dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=512,
        )

    print("Tokenizing dataset...")
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM, not masked LM
    )

    # VERY small training config, just to see it run.
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=5e-5,
        weight_decay=0.0,
        logging_steps=1,
        save_strategy="epoch",
        bf16=False,
        fp16=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving fine-tuned model to: {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("Done.")


if __name__ == "__main__":
    main()

