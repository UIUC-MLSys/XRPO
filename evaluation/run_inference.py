import os
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Any
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text from parquet datasets using vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g., 'Qwen/Qwen3-1.7B')"
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--input-parquets",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to input parquet files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save output JSON files"
    )
    
    # Generation parameters
    parser.add_argument(
        "--num-generations",
        type=int,
        default=5,
        help="Number of generations per prompt"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens to generate"
    )
    
    # GPU settings
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help="Number of GPUs to use for tensor parallelism"
    )
    
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization fraction"
    )
    
    # Additional options
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=True,
        help="Enable thinking mode in chat template"
    )

    return parser.parse_args()


def load_and_prepare_prompts(
    parquet_path: str,
    tokenizer: AutoTokenizer,
    enable_thinking: bool = True
) -> tuple[pd.DataFrame, List[str]]:
    """Load parquet file and prepare prompts with chat template."""
    print(f"Loading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    print(f"Loaded {len(df)} rows from {parquet_path}")
    
    # Apply chat template to prompts
    prompts = df["prompt"].apply(
        lambda x: tokenizer.apply_chat_template(
            x, 
            tokenize=False, 
            add_generation_prompt=True, 
            enable_thinking=enable_thinking
        )
    ).tolist()
    
    return df, prompts


def generate_outputs(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams
) -> List[List[str]]:
    """Generate outputs for prompts."""
    outputs = llm.generate(prompts, sampling_params)
    gen_outputs = [
        [output.outputs[idx].text for idx in range(sampling_params.n)]
        for output in outputs
    ]
    
    return gen_outputs


def save_generations(
    original_prompts: List[Any],
    generations: List[List[str]],
    output_path: str,
) -> None:
    """Save generations as JSON with prompts as keys."""
    # Create output dictionary with prompts as keys
    output_dict = {}
    
    for idx, (orig_prompt, gens) in enumerate(
        zip(original_prompts, generations)
    ):
        # Use the formatted prompt as the key
        key = str(orig_prompt)
        
        # Store original prompt and generations
        output_dict[key] = {
            "generations": list(gens),
            "index": idx
        }
    
    # Save as JSON
    print(f"Saving generations to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f)


def main():
    """Main function."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tokenizer
    print(f"Loading tokenizer for model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Initialize LLM
    print(f"Initializing LLM with tensor_parallel_size={args.tensor_parallel_size}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # Setup sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.num_generations,
        max_tokens=args.max_tokens
    )
    
    # Process each parquet file independently
    for parquet_path in args.input_parquets:
        try:
            # Load and prepare prompts
            df, formatted_prompts = load_and_prepare_prompts(
                parquet_path,
                tokenizer,
                args.enable_thinking
            )
            
            # Get original prompts
            original_prompts = df["prompt"].tolist()
            
            # Generate outputs
            print(f"Generating {args.num_generations} outputs per prompt...")
            gen_outputs = generate_outputs(
                llm,
                formatted_prompts,
                sampling_params
            )
            

            data_source = df["data_source"].iloc[0]
            base_name = Path(data_source).name
            
            output_filename = f"{base_name}_{Path(args.model).name}_generations.json"
            output_path = os.path.join(args.output_dir, output_filename)
            
            # Save generations
            save_generations(
                original_prompts,
                gen_outputs,
                output_path
            )
            
            print(f"Successfully processed {parquet_path}")
            print(f"Output saved to {output_path}")
            print("-" * 80)
            
        except Exception as e:
            print(f"Error processing {parquet_path}: {e}")
            continue


if __name__ == "__main__":
    main()
