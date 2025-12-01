"""
Local LLM-powered flight data analysis using Llama 3.2
Loads flight data and uses a locally-running Hugging Face model for analysis
"""

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
from huggingface_hub import login

def authenticate_huggingface():
    """
    Authenticate with Hugging Face using token from environment variable
    """
    hf_token = os.environ.get('HUGGINGFACE_TOKEN')
    if not hf_token:
        print("ERROR: HUGGINGFACE_TOKEN environment variable not set", file=sys.stderr)
        print("You need a Hugging Face token to access Meta's Llama models", file=sys.stderr)
        print("Get one from: https://huggingface.co/settings/tokens", file=sys.stderr)
        sys.exit(1)
    
    print("Authenticating with Hugging Face...")
    login(token=hf_token, new_session=True)
    print("Authentication successful!")

def load_model(model_name="meta-llama/Llama-3.2-1B-Instruct"):
    """
    Load the Hugging Face model and tokenizer
    
    Args:
        model_name: Name of the model to load from Hugging Face
    
    Returns:
        tuple: (tokenizer, model)
    """
    print(f"Loading model: {model_name}")
    print("This may take a few minutes on first run...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print(f"Model loaded successfully!")
    print(f"Model device: {model.device}")
    
    return tokenizer, model

def load_and_process_data(data_path):
    """
    Load CSV and sort by count
    
    Args:
        data_path: Path to CSV file (can be local or GCS path)
    
    Returns:
        DataFrame with top 5 rows sorted by count
    """
    print(f"Loading data from {data_path}...")
    
    # Handle both local files and GCS paths
    if data_path.startswith('gs://'):
        # For GCS, pandas can read directly with gcsfs installed
        summary = pd.read_csv(data_path)
    else:
        summary = pd.read_csv(data_path)
    
    summary = summary.sort_values("count", ascending=False)
    top_routes = summary.head()
    
    print(f"Loaded {len(summary)} total routes")
    print(f"\nTop 5 routes:")
    print(top_routes)
    
    return top_routes

def analyze_with_local_llm(data, tokenizer, model, max_tokens=250):
    """
    Analyze data using locally-running LLM
    
    Args:
        data: DataFrame containing flight route data
        tokenizer: Hugging Face tokenizer
        model: Hugging Face model
        max_tokens: Maximum tokens to generate
    
    Returns:
        String containing LLM analysis
    """
    print("\nGenerating analysis with local LLM...")
    
    # Create the prompt
    prompt = f"""The following includes some of the most common international flight routes within our airline. Please only list the international routes and briefly suggest what this means for our team's short term operations: {data}"""
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Tokenize and prepare inputs
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    # Generate response
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    
    # Decode only the generated tokens (skip the input prompt)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    
    return response

def main():
    """Main execution function"""
    
    print("=" * 80)
    print("Flight Route Analysis with Local Llama 3.2 LLM")
    print("=" * 80)
    
    try:
        # Configuration from environment variables
        data_path = os.environ.get(
            'DATA_PATH',
            'gs://insert/bucketpath/here/2015-summary.csv'
        )
        model_name = os.environ.get(
            'MODEL_NAME',
            'meta-llama/Llama-3.2-1B-Instruct'
        )
        max_tokens = int(os.environ.get('MAX_TOKENS', '250'))
        
        # Step 1: Authenticate with Hugging Face
        authenticate_huggingface()
        
        # Step 2: Load the LLM model
        tokenizer, model = load_model(model_name)
        
        # Step 3: Load and process data
        top_routes = load_and_process_data(data_path)
        
        # Step 4: Analyze with LLM
        analysis = analyze_with_local_llm(top_routes, tokenizer, model, max_tokens)
        
        # Display results
        print("\n" + "=" * 80)
        print("LLM ANALYSIS")
        print("=" * 80)
        print(analysis)
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
