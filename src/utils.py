"""
Utility functions for data processing and model operations.
"""

import json
from typing import Dict, List


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load data from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """
    Save data to JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Output file path
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def format_conversation(messages: List[Dict]) -> str:
    """
    Format conversation messages into a readable string.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        
    Returns:
        Formatted conversation string
    """
    formatted = []
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


def print_trainable_parameters(model):
    """
    Print the number of trainable parameters in the model.
    
    Args:
        model: PyTorch model
    """
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    percentage = 100 * trainable_params / all_params if all_params > 0 else 0
    
    print(f"Trainable params: {trainable_params:,}")
    print(f"All params: {all_params:,}")
    print(f"Trainable %: {percentage:.4f}%")


def validate_jsonl_format(file_path: str) -> bool:
    """
    Validate that a JSONL file has the correct format for training.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        data = load_jsonl(file_path)
        
        if not data:
            print(f"Error: {file_path} is empty")
            return False
        
        for i, item in enumerate(data):
            if "messages" not in item:
                print(f"Error: Line {i+1} missing 'messages' field")
                return False
            
            messages = item["messages"]
            if not isinstance(messages, list) or len(messages) == 0:
                print(f"Error: Line {i+1} 'messages' must be a non-empty list")
                return False
            
            for j, msg in enumerate(messages):
                if "role" not in msg or "content" not in msg:
                    print(f"Error: Line {i+1}, message {j+1} missing 'role' or 'content'")
                    return False
                
                if msg["role"] not in ["user", "assistant", "system"]:
                    print(f"Warning: Line {i+1}, message {j+1} has invalid role: {msg['role']}")
        
        print(f"✓ {file_path} is valid ({len(data)} examples)")
        return True
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"Error validating {file_path}: {e}")
        return False


def create_sample_dataset(output_path: str, num_samples: int = 10):
    """
    Create a sample dataset for testing.
    
    Args:
        output_path: Path to save sample dataset
        num_samples: Number of samples to generate
    """
    samples = []
    
    topics = [
        ("Python", "Python is a high-level programming language..."),
        ("Machine Learning", "Machine learning is a subset of AI..."),
        ("Data Science", "Data science involves extracting insights..."),
        ("Neural Networks", "Neural networks are computing systems..."),
        ("Deep Learning", "Deep learning uses multiple layers..."),
    ]
    
    for i in range(num_samples):
        topic, description = topics[i % len(topics)]
        sample = {
            "messages": [
                {"role": "user", "content": f"Tell me about {topic}"},
                {"role": "assistant", "content": description},
            ]
        }
        samples.append(sample)
    
    save_jsonl(samples, output_path)
    print(f"Created sample dataset with {num_samples} examples: {output_path}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset utilities")
    parser.add_argument("--validate", type=str, help="Validate JSONL file")
    parser.add_argument("--create-sample", type=str, help="Create sample dataset")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples")
    
    args = parser.parse_args()
    
    if args.validate:
        validate_jsonl_format(args.validate)
    
    if args.create_sample:
        create_sample_dataset(args.create_sample, args.num_samples)
