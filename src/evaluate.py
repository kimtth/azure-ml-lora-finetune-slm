"""
Evaluation script for fine-tuned language models.

This script evaluates fine-tuned models on various metrics including:
- Perplexity
- ROUGE scores
- Generation quality
- Inference latency
"""

import os
import json
import time
import argparse
from typing import Dict, List

import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from rouge_score import rouge_scorer
from tqdm import tqdm
import mlflow


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LLM")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model (LoRA adapter)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/Phi-4-mini-instruct",
        help="Base model name or path (should match training config)",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test data (JSONL format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=False,
        help="Load model in 4-bit precision",
    )
    parser.add_argument(
        "--mlflow_tracking",
        action="store_true",
        default=True,
        help="Enable MLflow tracking",
    )
    
    return parser.parse_args()


def load_jsonl_dataset(file_path: str, max_samples: int = None) -> Dataset:
    """Load dataset from JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data.append(json.loads(line))
    return Dataset.from_list(data)


def load_model_and_tokenizer(args):
    """Load fine-tuned model and tokenizer."""
    print("Loading model and tokenizer...")
    
    # Configure quantization if needed
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="left",  # For batch inference
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.load_in_4bit else torch.float16,
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()
    
    print(f"Model loaded from: {args.model_path}")
    return model, tokenizer


def calculate_perplexity(model, tokenizer, dataset: Dataset) -> float:
    """Calculate perplexity on the dataset."""
    print("\nCalculating perplexity...")
    
    total_loss = 0
    total_tokens = 0
    
    model.eval()
    with torch.no_grad():
        for example in tqdm(dataset, desc="Computing perplexity"):
            messages = example["messages"]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(model.device)
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return perplexity


def generate_responses(
    model,
    tokenizer,
    dataset: Dataset,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[Dict]:
    """Generate responses for the dataset."""
    print("\nGenerating responses...")
    
    results = []
    model.eval()
    
    for example in tqdm(dataset, desc="Generating"):
        messages = example["messages"]
        
        # Extract user message and reference response
        user_message = None
        reference_response = None
        
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                user_message = msg["content"]
            elif msg["role"] == "assistant" and user_message:
                reference_response = msg["content"]
                break
        
        if not user_message:
            continue
        
        # Prepare prompt
        prompt_messages = [{"role": "user", "content": user_message}]
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        generation_time = time.time() - start_time
        
        # Decode response
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        results.append({
            "user_message": user_message,
            "reference_response": reference_response,
            "generated_response": generated_text,
            "generation_time": generation_time,
        })
    
    return results


def calculate_rouge_scores(results: List[Dict]) -> Dict[str, float]:
    """Calculate ROUGE scores."""
    print("\nCalculating ROUGE scores...")
    
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    
    rouge_scores = {
        "rouge1_f": [],
        "rouge2_f": [],
        "rougeL_f": [],
    }
    
    for result in results:
        if result["reference_response"]:
            scores = scorer.score(
                result["reference_response"],
                result["generated_response"],
            )
            
            rouge_scores["rouge1_f"].append(scores["rouge1"].fmeasure)
            rouge_scores["rouge2_f"].append(scores["rouge2"].fmeasure)
            rouge_scores["rougeL_f"].append(scores["rougeL"].fmeasure)
    
    # Calculate averages
    avg_scores = {
        key: np.mean(values) for key, values in rouge_scores.items()
    }
    
    return avg_scores


def calculate_latency_metrics(results: List[Dict]) -> Dict[str, float]:
    """Calculate latency metrics."""
    generation_times = [r["generation_time"] for r in results]
    
    return {
        "avg_latency": np.mean(generation_times),
        "median_latency": np.median(generation_times),
        "p95_latency": np.percentile(generation_times, 95),
        "p99_latency": np.percentile(generation_times, 99),
    }


def save_results(results: List[Dict], metrics: Dict, output_dir: str):
    """Save evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(output_dir, "detailed_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save metrics summary
    metrics_file = os.path.join(output_dir, "metrics_summary.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Start MLflow run
    if args.mlflow_tracking:
        mlflow.start_run()
        mlflow.log_params({
            "model_path": args.model_path,
            "base_model": args.base_model,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        })
    
    print("=" * 80)
    print("Starting Model Evaluation")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Test data: {args.test_data}")
    print("=" * 80)
    
    # Load model and tokenizer
    print("\n[1/5] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Load test dataset
    print("\n[2/5] Loading test dataset...")
    test_dataset = load_jsonl_dataset(args.test_data, args.max_samples)
    print(f"Test examples: {len(test_dataset)}")
    
    # Calculate perplexity
    print("\n[3/5] Calculating perplexity...")
    perplexity = calculate_perplexity(model, tokenizer, test_dataset)
    
    # Generate responses
    print("\n[4/5] Generating and evaluating responses...")
    results = generate_responses(
        model,
        tokenizer,
        test_dataset,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )
    
    # Calculate metrics
    print("\n[5/5] Computing metrics...")
    rouge_scores = calculate_rouge_scores(results)
    latency_metrics = calculate_latency_metrics(results)
    
    # Compile all metrics
    all_metrics = {
        "perplexity": perplexity,
        **rouge_scores,
        **latency_metrics,
        "num_samples": len(results),
    }
    
    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Perplexity: {perplexity:.4f}")
    print("\nROUGE Scores:")
    print(f"  ROUGE-1 F1: {rouge_scores['rouge1_f']:.4f}")
    print(f"  ROUGE-2 F1: {rouge_scores['rouge2_f']:.4f}")
    print(f"  ROUGE-L F1: {rouge_scores['rougeL_f']:.4f}")
    print("\nLatency Metrics:")
    print(f"  Average: {latency_metrics['avg_latency']:.4f}s")
    print(f"  Median: {latency_metrics['median_latency']:.4f}s")
    print(f"  P95: {latency_metrics['p95_latency']:.4f}s")
    print(f"  P99: {latency_metrics['p99_latency']:.4f}s")
    print("=" * 80)
    
    # Save results
    save_results(results, all_metrics, args.output_dir)
    
    # Log to MLflow
    if args.mlflow_tracking:
        mlflow.log_metrics(all_metrics)
        mlflow.log_artifact(args.output_dir)
        mlflow.end_run()
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
