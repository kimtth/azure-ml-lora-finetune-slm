"""
Compare base model vs fine-tuned model outputs on sample queries.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Sample test queries
TEST_QUERIES = [
    "What is machine learning?",
    "Explain the difference between supervised and unsupervised learning.",
    "How do neural networks work?",
    "What are the benefits of cloud computing?",
    "Describe the concept of natural language processing.",
]


def load_base_model(model_name: str):
    """Load base model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    return model, tokenizer


def load_finetuned_model(base_model_name: str, adapter_path: str):
    """Load fine-tuned model with LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model, tokenizer


def generate_response(model, tokenizer, query: str, max_tokens: int = 256) -> str:
    """Generate response for a query."""
    messages = [{"role": "user", "content": query}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("assistant")[-1].strip() if "assistant" in response else response


def compare_models(base_model_name: str, adapter_path: str, queries: list):
    """Compare base and fine-tuned models."""
    print("=" * 100)
    print("LOADING MODELS")
    print("=" * 100)
    
    print("\n[1/2] Loading base model...")
    base_model, base_tokenizer = load_base_model(base_model_name)
    
    print("[2/2] Loading fine-tuned model...")
    ft_model, ft_tokenizer = load_finetuned_model(base_model_name, adapter_path)
    
    print("\n" + "=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*100}")
        print(f"Query {i}: {query}")
        print("=" * 100)
        
        print("\n[BASE MODEL]")
        base_response = generate_response(base_model, base_tokenizer, query)
        print(base_response)
        
        print("\n[FINE-TUNED MODEL]")
        ft_response = generate_response(ft_model, ft_tokenizer, query)
        print(ft_response)
        print()
    
    print("=" * 100)
    print("COMPARISON COMPLETE")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/Phi-4-mini-instruct",
        help="Base model name",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to fine-tuned LoRA adapters",
    )
    parser.add_argument(
        "--queries",
        type=str,
        nargs="+",
        default=None,
        help="Custom queries to test",
    )
    
    args = parser.parse_args()
    queries = args.queries if args.queries else TEST_QUERIES
    
    compare_models(args.base_model, args.adapter_path, queries)


if __name__ == "__main__":
    main()
