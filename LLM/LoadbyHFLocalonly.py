import os

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

N_THREADS = os.cpu_count() - 2 if os.cpu_count() > 2 else 1  # Reserve cores for OS

def load_llm(param):
    if not param:
        print("No input parameter for HuggingFace model. Check configuration.")
        return None
    
    model, token, context_size = param
    context_size = int(context_size) if context_size else 2048
    print("model {model}, token{token}, context_size {context_size}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True
        )
        
        # Load model with CPU optimization
        model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float32,  # Use float32 for CPU
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
        
        print("Model and tokenizer loaded successfully")
        
        # Create pipeline for text generation
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
    except Exception as e:
        print(f"Failed to load HuggingFace model: {e}")
        return None