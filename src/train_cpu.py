"""
CPU-optimized fine-tuning script for Small Language Models using LoRA.
Designed for non-GPU Azure ML compute instances.
"""

import os
import argparse
from typing import Dict, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from trl import SFTTrainer, SFTConfig
import mlflow


def parse_args():
    parser = argparse.ArgumentParser(description="CPU-optimized LoRA fine-tuning")
    
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-4-mini-instruct")
    parser.add_argument("--dataset_name", type=str, default="databricks/databricks-dolly-15k")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--validation_size", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=1000, help="Limit dataset size for CPU")
    parser.add_argument("--max_seq_length", type=int, default=256)
    
    parser.add_argument("--lora_r", type=int, default=8, help="Reduced for CPU")
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, nargs="+", default=["q_proj", "v_proj"])
    
    parser.add_argument("--output_dir", type=str, default="./outputs_cpu")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow_tracking", action="store_true", default=True)
    
    return parser.parse_args()


def load_and_prepare_dataset(dataset_name: str, train_split: str, validation_size: float, max_samples: Optional[int] = None):
    dataset = load_dataset(dataset_name, split=train_split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    split_dataset = dataset.train_test_split(test_size=validation_size, seed=42)
    return split_dataset["train"], split_dataset["test"]


def format_dolly_to_chat(example: Dict, tokenizer) -> Dict:
    user_content = example['instruction']
    if example.get('context'):
        user_content += f"\n\n{example['context']}"
    
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example['response']}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def setup_model_and_tokenizer_cpu(args):
    """Setup model and tokenizer for CPU training (no quantization)."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("Loading model on CPU (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,  # CPU uses float32
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    
    if args.mlflow_tracking:
        mlflow.start_run()
        mlflow.log_params({
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "max_samples": args.max_samples,
            "lora_r": args.lora_r,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "compute_type": "CPU",
        })
    
    print("=" * 80)
    print("CPU-Optimized LoRA Fine-tuning")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Dataset samples: {args.max_samples}")
    print(f"LoRA rank: {args.lora_r}")
    print("Device: CPU (no quantization)")
    print("=" * 80)
    
    print("\n[1/5] Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer_cpu(args)
    
    print("\n[2/5] Loading datasets...")
    train_dataset, eval_dataset = load_and_prepare_dataset(
        args.dataset_name, args.train_split, args.validation_size, args.max_samples
    )
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")
    
    print("\n[3/5] Formatting datasets...")
    train_dataset = train_dataset.map(
        lambda x: format_dolly_to_chat(x, tokenizer),
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda x: format_dolly_to_chat(x, tokenizer),
        remove_columns=eval_dataset.column_names,
    )
    
    print("\n[4/5] Configuring CPU training...")
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        save_strategy="steps",
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=False,  # No mixed precision on CPU
        bf16=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        max_seq_length=args.max_seq_length,
        packing=False,
        dataset_text_field="text",
        report_to="mlflow" if args.mlflow_tracking else "none",
        seed=args.seed,
        dataloader_num_workers=2,  # Use 2 cores for data loading
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    print("\n[5/5] Starting training...")
    print("=" * 80)
    print("NOTE: CPU training is slower than GPU. Estimated time:")
    print(f"  ~{len(train_dataset) * args.num_epochs // (args.batch_size * args.gradient_accumulation_steps) * 10} seconds")
    print("=" * 80)
    
    trainer.train()
    
    print("\n" + "=" * 80)
    print("Training completed! Saving model...")
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model saved to: {final_model_path}")
    
    if args.mlflow_tracking:
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)
        mlflow.end_run()
    
    print("=" * 80)
    print("CPU fine-tuning complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
