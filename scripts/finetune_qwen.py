"""
finetune_qwen.py -- Phase C: QLoRA fine-tune of Qwen2.5-Coder-7B-Instruct on the
enterprise text-to-SQL corpus (data/ft_train.jsonl from make_finetune_data.py).

Why this over the from-scratch model: the backbone already understands language
and SQL, so it generalizes across phrasings the 317 templates never taught.
QLoRA (4-bit base + small LoRA adapters) fits a single 16GB GPU (your T4),
and 4-bit inference runs in ~6-8GB on-prem -- fully local, data never leaves.

Deps (pin to avoid trl/peft API drift -- the main risk in this file):
  pip install "transformers>=4.45" "trl>=0.12,<0.15" "peft>=0.13" \
              "bitsandbytes>=0.44" "accelerate>=1.0" datasets

Run:
  python scripts/finetune_qwen.py
Output: weights/qwen_sql_lora/  (LoRA adapter; base stays on HF cache)

NOTE: untested locally (needs a 16GB+ GPU). First cloud run is the smoke test --
watch that loss drops and eval_loss tracks it. If trl's API differs in your
version, the SFTConfig field names are the usual culprit (see comments).
"""
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 14B is the quality jump worth spending the budget on; fits 16GB in 4-bit.
# If it OOMs, fall back to "Qwen/Qwen2.5-Coder-7B-Instruct" (one line).
MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# 4-bit NF4 quantization (QLoRA): base weights frozen in 4-bit, LoRA trains in bf16.
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

lora = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

ds = load_dataset("json", data_files={
    "train": os.path.join(BASE, "data", "ft_train.jsonl"),
    "val":   os.path.join(BASE, "data", "ft_val.jsonl"),
})

# Render each {"messages":[...]} into a single chat-formatted string. Stable
# across trl versions (we pass dataset_text_field rather than relying on the
# trainer's internal chat handling).
def render(ex):
    return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}
ds = ds.map(render, remove_columns=ds["train"].column_names)

# Completion-only loss: mask the system+user prompt, train only on the SQL the
# model should produce. Qwen's chat template starts the assistant turn with this
# marker, so loss begins right after it.
collator = DataCollatorForCompletionOnlyLM(
    response_template="<|im_start|>assistant\n", tokenizer=tok)

cfg = SFTConfig(
    output_dir=os.path.join(BASE, "weights", "qwen_sql_lora"),
    num_train_epochs=2,                 # narrow domain -> 2-3 is plenty
    per_device_train_batch_size=1,      # 14B: bs1 x accum16 = eff batch 16
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",              # older trl: evaluation_strategy
    load_best_model_at_end=True,        # keep best eval_loss epoch -> budget-safe
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    # MUST exceed system(~2.2k) + question + SQL, else the answer (which comes
    # last) is truncated and the completion-only collator masks everything ->
    # zero training signal. 3072 leaves comfortable room.
    max_seq_length=3072,
    dataset_text_field="text",          # lives in SFTConfig in current trl
    packing=False,
    report_to="none",
)

# NOTE: in current trl, dataset_text_field / max_seq_length belong to SFTConfig
# (above), NOT to SFTTrainer(...). Passing them to the trainer raises
# "unexpected keyword argument".
trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    peft_config=lora,
    data_collator=collator,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(os.path.join(BASE, "weights", "qwen_sql_lora"))
    tok.save_pretrained(os.path.join(BASE, "weights", "qwen_sql_lora"))
    print("Saved LoRA adapter -> weights/qwen_sql_lora")
