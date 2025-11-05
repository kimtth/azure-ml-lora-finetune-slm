"""
Fine-tuning script for Small Language Models using LoRA on Azure ML.

This script fine-tunes open-source language models using Parameter-Efficient Fine-Tuning (PEFT)
with LoRA adapters and 4-bit quantization for reduced memory footprint.
"""

import os
import argparse
from typing import Dict, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig
import mlflow


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune LLM with LoRA")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Phi-4-mini-instruct",
        help="Pretrained model name or path",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Tokenizer name (defaults to model_name)",
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="databricks/databricks-dolly-15k",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.05,
        help="Validation split size",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples to use (for testing)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Target modules for LoRA",
    )
    
    # Quantization arguments
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=True,
        help="Load model in 4-bit precision",
    )
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        type=str,
        default="bfloat16",
        help="Compute dtype for 4-bit quantization",
    )
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing",
    )
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # MLflow tracking
    parser.add_argument(
        "--mlflow_tracking",
        action="store_true",
        default=True,
        help="Enable MLflow tracking",
    )
    
    return parser.parse_args()


def load_and_prepare_dataset(dataset_name: str, train_split: str, validation_size: float, max_samples: Optional[int] = None):
    """Load dataset from HuggingFace and split into train/validation."""
    dataset = load_dataset(dataset_name, split=train_split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    split_dataset = dataset.train_test_split(test_size=validation_size, seed=42)
    return split_dataset["train"], split_dataset["test"]


def format_dolly_to_chat(example: Dict, tokenizer) -> Dict:
    """Convert Dolly format to chat template."""
    user_content = example['instruction']
    if example.get('context'):
        user_content += f"\n\n{example['context']}"
    
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example['response']}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def setup_model_and_tokenizer(args):
    """Set up model and tokenizer with quantization and LoRA."""
    # Configure 4-bit quantization
    if args.load_in_4bit:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        bnb_config = None
    
    # Load tokenizer
    tokenizer_name = args.tokenizer_name or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.load_in_4bit else torch.float16,
    )
    
    # Prepare model for k-bit training
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Start MLflow run
    if args.mlflow_tracking:
        mlflow.start_run()
        mlflow.log_params({
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length,
        })
    
    print("=" * 80)
    print("Starting LoRA Fine-tuning")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"4-bit quantization: {args.load_in_4bit}")
    print("=" * 80)
    
    # Load model and tokenizer
    print("\n[1/5] Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(args)
    
    # Load datasets
    print("\n[2/5] Loading datasets...")
    train_dataset, eval_dataset = load_and_prepare_dataset(
        args.dataset_name, args.train_split, args.validation_size, args.max_samples
    )
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")
    
    # Format datasets
    print("\n[3/5] Formatting datasets...")
    train_dataset = train_dataset.map(
        lambda x: format_dolly_to_chat(x, tokenizer),
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda x: format_dolly_to_chat(x, tokenizer),
        remove_columns=eval_dataset.column_names,
    )
    
    # Configure training
    print("\n[4/5] Configuring training...")
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        save_strategy="steps",
        eval_strategy="steps" if eval_dataset else "no",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="loss" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        max_seq_length=args.max_seq_length,
        packing=False,
        dataset_text_field="text",
        report_to="mlflow" if args.mlflow_tracking else "none",
        seed=args.seed,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # Train model
    print("\n[5/5] Starting training...")
    print("=" * 80)
    trainer.train()
    
    # Save final model
    print("\n" + "=" * 80)
    print("Training completed! Saving model...")
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"Model saved to: {final_model_path}")
    
    # Log final metrics
    if args.mlflow_tracking:
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(final_model_path)
        mlflow.end_run()
    
    print("=" * 80)
    print("Fine-tuning complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
