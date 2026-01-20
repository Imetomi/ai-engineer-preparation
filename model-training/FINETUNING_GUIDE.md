# Fine-Tuning Guide for Qwen3-VL:2B-Instruct

## What is Fine-Tuning?

**Fine-tuning** = taking a pre-trained model and training it further on your specific data to improve performance on your task.

### Analogy
- **Pre-training**: Model learns general language (like learning to read/write in school)
- **Fine-tuning**: Model learns your specific style/task (like learning medical terminology for a doctor)

### How It Works
1. Start with a pre-trained model (e.g., Qwen base model)
2. Load your small dataset (50-500 examples is fine for practice)
3. Train for a few epochs (1-3 usually enough)
4. Model adjusts its weights to better match your examples
5. Save the fine-tuned model

## Practical Approach for Your Model

### Challenge: Ollama Models Are Quantized
- Your `qwen3-vl:2b-instruct` in Ollama is a **quantized GGUF** file
- You **can't directly fine-tune** a quantized model
- **Solution**: Fine-tune the **base model** first, then convert to GGUF for Ollama

### Recommended Libraries

#### Option 1: Unsloth (Easiest for Beginners)
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" trl peft accelerate bitsandbytes
```

**Why Unsloth?**
- Super fast (2-5x faster than standard training)
- Easy API, minimal code
- Built-in LoRA support (trains only a small % of parameters)
- Works great with small models like 2B

#### Option 2: Transformers + PEFT (More Control)
```bash
pip install transformers peft accelerate bitsandbytes datasets
```

**Why PEFT?**
- LoRA (Low-Rank Adaptation): only trains ~1-5% of parameters
- Much faster, less memory, often better results
- Standard HuggingFace approach

## Dataset Format

### For Instruction Tuning (What You Need)

Your dataset should be a **JSON file** with this structure:

```json
[
  {
    "instruction": "Answer based on the provided context from my personal notes.",
    "input": "Question: How should I handle my dreams about Rebi?\n\nContext: [your context here]",
    "output": "Based on your notes, you should..."
  },
  {
    "instruction": "Answer based on the provided context from my personal notes.",
    "input": "Question: What is GraphRAG?\n\nContext: [your context here]",
    "output": "GraphRAG is..."
  }
]
```

### Creating Your Dataset

**Small practice dataset (50-100 examples):**
1. Take 20-30 real questions from your GraphRAG queries
2. For each question, manually write a good answer (or use your current agent to generate, then edit)
3. Include the context that was used (from your semantic + graph search)
4. Format as JSON above

**Example script to generate dataset from your queries:**

```python
# backend/generate_finetune_dataset.py
import json
from agent import query
from ingest import load_markdown_notes

# Run a bunch of queries, collect Q&A pairs
dataset = []

questions = [
    "How should I handle my dreams about a person?",
    "What is GraphRAG?",
    "Explain how Neo4j works with my notes.",
    # ... more questions
]

for q in questions:
    answer = query(q)
    # You'd manually review/edit answers, then add to dataset
    dataset.append({
        "instruction": "Answer based on the provided context from my personal notes.",
        "input": f"Question: {q}",
        "output": answer  # or your edited version
    })

with open("finetune_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)
```

## Step-by-Step Fine-Tuning with Unsloth

### 1. Install Dependencies
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" trl peft accelerate bitsandbytes
```

### 2. Load Base Model (Not the Ollama Quantized One)
```python
from unsloth import FastLanguageModel
import torch

# Load the BASE Qwen model (not the quantized one)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-2B-Instruct",  # Base model from HuggingFace
    max_seq_length = 2048,
    dtype = None,  # Auto detection
    load_in_4bit = True,  # 4-bit quantization for memory efficiency
)

# Add LoRA adapters (only trains ~1% of parameters)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
```

### 3. Load Your Dataset
```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="finetune_dataset.json", split="train")

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}<|im_end|>\n"
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
```

### 4. Train
```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 3,  # 1-3 is usually enough
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",
        seed = 3407,
    ),
)

# Train!
trainer.train()
```

### 5. Save Model
```python
# Save LoRA adapters (small, ~10-50MB)
model.save_pretrained("lora_model")

# Or merge and save full model
FastLanguageModel.for_inference(model)  # Merge adapters
model.save_pretrained_merged("merged_model", tokenizer, save_method="merged_16bit")
```

### 6. Convert to GGUF for Ollama (Optional)
```bash
# Use llama.cpp to convert
python -m llama_cpp.convert \
    --outfile qwen2.5-2b-finetuned.gguf \
    --outtype f16 \
    merged_model/
```

## Alternative: Quick "Fine-Tuning" with Ollama Modelfile

If you just want to **practice** without full fine-tuning, Ollama supports **few-shot examples** via Modelfile:

```dockerfile
# Create a Modelfile
FROM qwen3-vl:2b-instruct

SYSTEM """You are a helpful assistant answering questions based on a personal knowledge graph built from Obsidian notes."""

TEMPLATE """{{ .System }}

{{ .Prompt }}"""

# Add few-shot examples
PARAMETER num_ctx 2048
```

Then create custom model:
```bash
ollama create my-custom-qwen -f Modelfile
```

**This isn't true fine-tuning** (doesn't change weights), but it's a quick way to customize behavior.

## Recommended Path for You

1. **Start simple**: Create a small dataset (20-50 examples) from your real queries
2. **Use Unsloth**: Easiest to get started, fast, works great with 2B models
3. **Train LoRA**: Only trains ~1% of params, fast, good results
4. **Test locally**: Load fine-tuned model, compare to base
5. **Convert to GGUF** (if you want to use with Ollama later)

## Resources

- **Unsloth docs**: https://github.com/unslothai/unsloth
- **Qwen models**: https://huggingface.co/Qwen
- **LoRA paper**: https://arxiv.org/abs/2106.09685
- **Ollama Modelfile**: https://github.com/ollama/ollama/blob/main/docs/modelfile.md
