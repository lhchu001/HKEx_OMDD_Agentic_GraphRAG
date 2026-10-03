# LoadbyHuggingFace.py
import os

from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEndpoint
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

N_THREADS = os.cpu_count() - 2 if os.cpu_count() > 2 else 1  # Reserve cores for OS

def load_llm(param):
    if not param:
        print("No input parameter for HuggingFace model. Check configuration.")
        return None
    
    model, token, context_size = param
    context_size = int(context_size) if context_size else 2048
    print(f"model: {model}, token: {'provided' if token else 'none'}, context_size: {context_size}")
    
    # Cloud inference if token is provided
    if token and token.strip():
        try:
            print("Loading model via Hugging Face cloud Inference API...")
            llm_endpoint = HuggingFaceEndpoint(
                repo_id=model,
                huggingfacehub_api_token=token,
                task="conversational",  # Keeps chat completions endpoint (Featherless AI)
                temperature=0.7,        # Explicit (required by validation)
                top_p=0.9,              # Explicit (required by validation)
                max_new_tokens=512,     # Moved out of model_kwargs to top-level
                model_kwargs={
                    # Omit repetition_penalty and do_sample (not standard in chat endpoints)
                }
            )

            chat_model = ChatHuggingFace(llm=llm_endpoint)

            print("Cloud model loaded successfully")
            return chat_model
        except Exception as e:
            print(f"Failed to load cloud model: {e}")
            print("Falling back to local loading...")
    
    # Local loading (fallback or default)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
            token=token if token else None
            )
        
        model_obj = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float32,  # Or use dtype=torch.float32 to avoid deprecation warning
            trust_remote_code=True,
            low_cpu_mem_usage=True
            # Remove device_map entirely
            )
        
        print("Local model and tokenizer loaded successfully")
        
        pipe = pipeline(
            "text-generation",
            model=model_obj,
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
        print(f"Failed to load local HuggingFace model: {e}")
        return None