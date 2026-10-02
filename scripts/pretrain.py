#!/usr/bin/env python3
"""Continued causal-language-model pretraining on a user-supplied dataset."""

import argparse
import math
from dataclasses import dataclass

from datasets import load_dataset, load_from_disk
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments, set_seed)


def dataset_arg(value: str, split: str):
    try:
        return load_from_disk(value)[split] if hasattr(load_from_disk(value), "keys") else load_from_disk(value)
    except (FileNotFoundError, ValueError, OSError):
        return load_dataset(value, split=split)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, help="Hugging Face dataset id or load_from_disk directory")
    p.add_argument("--dataset-config")
    p.add_argument("--train-split", default="train")
    p.add_argument("--validation-split", default="validation")
    p.add_argument("--text-column", default="text")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--block-size", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--max-train-samples", type=int)
    return p.parse_args()


def main():
    a = parse_args()
    set_seed(a.seed)
    try:
        train = dataset_arg(a.dataset, a.train_split)
        valid = dataset_arg(a.dataset, a.validation_split)
    except Exception as exc:
        raise SystemExit(f"Could not load dataset/split: {exc}") from exc
    if a.max_train_samples:
        train = train.select(range(min(a.max_train_samples, len(train))))
    if a.text_column not in train.column_names:
        raise SystemExit(f"Missing --text-column={a.text_column}; available: {train.column_names}")

    tokenizer = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model)
    model.config.use_cache = False

    def tokenize(batch):
        return tokenizer(batch[a.text_column], add_special_tokens=True)

    remove = [c for c in train.column_names if c != a.text_column]
    train = train.map(tokenize, batched=True, remove_columns=remove)
    valid = valid.map(tokenize, batched=True, remove_columns=[c for c in valid.column_names if c != a.text_column])

    def pack(batch):
        joined = {k: sum(batch[k], []) for k in batch}
        n = (len(joined["input_ids"]) // a.block_size) * a.block_size
        return {k: [v[i : i + a.block_size] for i in range(0, n, a.block_size)] for k, v in joined.items()}

    train, valid = train.map(pack, batched=True), valid.map(pack, batched=True)
    args = TrainingArguments(
        output_dir=a.output_dir, learning_rate=a.learning_rate, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.gradient_accumulation_steps, gradient_checkpointing=True,
        warmup_ratio=0.03, weight_decay=0.1, logging_steps=10, eval_strategy="steps",
        eval_steps=a.eval_steps, save_steps=a.save_steps, save_total_limit=2,
        bf16=True, report_to="none", prediction_loss_only=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=valid,
                      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    trainer.train()
    metrics = trainer.evaluate()
    if "eval_loss" in metrics:
        metrics["perplexity"] = math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 20 else float("inf")
    trainer.save_model(a.output_dir)
    tokenizer.save_pretrained(a.output_dir)
    print(metrics)


if __name__ == "__main__":
    main()
