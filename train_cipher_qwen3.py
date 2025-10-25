from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# === Base model ===
model_name = "Qwen/Qwen3-8B"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)

# === LoRA Configuration ===
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)

# === Load dataset ===
print("Loading dataset...")
dataset = load_dataset("json", data_files="datasets/cipher_data.jsonl")["train"]

def tokenize(example):
    text = f"Instruction:\n{example['instruction']}\nResponse:\n{example['response']}"
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=1024)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


tokenized = dataset.map(tokenize)

# === Training parameters ===
args = TrainingArguments(
    output_dir="cipher_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=20,
    optim="paged_adamw_8bit",
    fp16=True,
    save_strategy="epoch"
)

print("Starting training...")
trainer = Trainer(model=model, args=args, train_dataset=tokenized)
trainer.train()

# === Save LoRA weights ===
print("Saving LoRA weights...")
model.save_pretrained("cipher_lora_trained")
print("Training complete ✅")
