import os

from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

N_THREADS = os.cpu_count() - 2 if os.cpu_count() > 2 else 1  # Reserve cores for OS

def load_llm(param):
    if not param:
        print(f"No input parameter, check the configuration.")
        return None

    model_path, context_size = param
    context_size = int(context_size)
    print(f"model_path {model_path}, context_size {context_size}")
	
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please download it first.")
        return None

    print(f"Loading model from {model_path}...")
 
    try:
        llm = LlamaCpp(
            model_path=model_path,
            n_ctx=context_size,
            n_threads=N_THREADS,
            temperature=0.1,  # Low temp for factual/tool usage
            max_tokens=512,
            top_p=0.9,
            verbose=False,    # Reduce C++ logs
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        return llm
    except Exception as e:
        print(f"Failed to load Llama.cpp: {e}")
        return None