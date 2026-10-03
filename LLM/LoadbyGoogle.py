from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model

def load_llm(param):
    if not param:
        print(f"No input parameter, check the configuration.")
        return None

    api_key, model, context_size = param
    context_size = int(context_size)
#    print(f"api_key {api_key}, model {model}, context_size {context_size}")
    print(f"model: {model}, context_size: {context_size}")

    # Initialize the Google GenAI model
    try:
#        llm = ChatGoogleGenerativeAI(
#               model=model,
#               google_api_key=api_key,
#               temperature=0,
#               )
        llm = init_chat_model(
                model=model,
                model_provider="google_genai",
                api_key=api_key,
                temperature=0,
                )
        return llm
    except Exception as e:
        print(f"Failed to load Google GenAI model: {e}")
        return None
