from langchain_openai import ChatOpenAI

def load_llm(param):
    if not param:
        print(f"No input parameter, check the configuration.")
        return None

    api_key, model, url, context_size = param
    context_size = int(context_size)
#    print(f"api_key {api_key}, model {model}, url {url}, context_size {context_size}")
 
    # Initialize the OpenRouter model
    # Note: base_url is set to OpenRouter's API endpoint
    try:
        llm = ChatOpenAI(
               model=model,
               openai_api_key=api_key,
               base_url=url,
               temperature=0,
               max_retries=3, # Add this to handle transient 429s
               )
        return llm
    except Exception as e:
        print(f"Failed to load Llama.cpp: {e}")
        return None