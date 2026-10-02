#!/usr/bin/env python3
"""Supervised instruction fine-tuning for a user-supplied text dataset."""

import argparse
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer


def load_any(path, split):
    try:
        loaded = load_from_disk(path)
        return loaded[split] if hasattr(loaded, "keys") else loaded
    except (FileNotFoundError, ValueError, OSError):
        return load_dataset(path, split=split)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--train-split", default="train")
    p.add_argument("--validation-split", default="validation")
    p.add_argument("--text-column", default="text")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora", action="store_true", help="Use a PEFT LoRA adapter")
    a = p.parse_args()
    set_seed(a.seed)
    train, valid = load_any(a.dataset, a.train_split), load_any(a.dataset, a.validation_split)
    if a.text_column not in train.column_names:
        raise SystemExit(f"Missing --text-column={a.text_column}; available: {train.column_names}")
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model)
    model.config.use_cache = False
    peft = None
    if a.lora:
        from peft import LoraConfig
        peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    cfg = SFTConfig(output_dir=a.output_dir, num_train_epochs=a.epochs, learning_rate=a.learning_rate,
                    per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size,
                    gradient_accumulation_steps=a.gradient_accumulation_steps, max_seq_length=a.max_seq_length,
                    packing=True, gradient_checkpointing=True, eval_strategy="steps", eval_steps=200,
                    save_steps=200, save_total_limit=2, bf16=True, report_to="none")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train, eval_dataset=valid,
                         processing_class=tok, dataset_text_field=a.text_column, peft_config=peft)
    trainer.train()
    trainer.save_model(a.output_dir)
    tok.save_pretrained(a.output_dir)


if __name__ == "__main__":
    main()
