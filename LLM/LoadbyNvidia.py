from langchain.chat_models import init_chat_model

"""
Load LLM Example - NVIDIA NIM Edition

This example demonstrates the LLM loading and environment setup for LangChain
using NVIDIA NIM endpoints.
"""

def load_llm(param):
    if not param:
        print(f"No input parameter, check the configuration.")
        return None

    api_key, model, context_size = param
    context_size = int(context_size)
 #   print(f"api_key {api_key}, model {model}, context_size {context_size}")
    print(f"model: {model}, context_size: {context_size}")
    
    # init_chat_model for NVIDIA
    try:
        # We use model_provider="nvidia" which maps to LangChain's ChatNVIDIA class
        model = init_chat_model(
            model=model, 
            model_provider="nvidia",
            api_key=api_key,
            temperature=0.5,
            max_tokens=1024
        )
        return model
    except Exception as e:
        print(f"Error initializing NVIDIA chat model: {e}")
        return None