import os

from langchain_ollama import ChatOllama

N_THREADS = os.cpu_count() - 2 if os.cpu_count() > 2 else 1  # Reserve cores for OS

def load_llm(param):
    """Loads an Ollama model with bind_tools() support."""
    if not param:
        print("No input parameter for Ollama model. Check configuration.")
        return None
    
    model, url, context_size = param
    context_size = int(context_size)
    print(f"model {model}, url: {url}, context_size {context_size}")
    
    try:
        # Initialize ChatOllama with tool calling support
        llm = ChatOllama(
            model=model,
            base_url=url,
            temperature=0.1,  # Low temperature for tool usage reliability
            num_ctx=context_size,     # Context window
            num_thread=N_THREADS,
            verbose=False
        )        
        return llm        
    except Exception as e:
        print(f"Failed to load Ollama model: {e}")
        return None