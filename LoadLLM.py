import os
import sys
import getpass
from pathlib import Path
from dotenv import load_dotenv

from LLM import LoadbyHuggingFace
from LLM import LoadbyOllama
from LLM import LoadbyOpenRouter
from LLM import LoadbyGoogle
from LLM import LoadbyLlamaCpp
from LLM import LoadbyNvidia


def load_config(env_path=None):
    # Get the directory where the script itself is located
    if (env_path is None):
        script_dir = Path(__file__).parent
        env_path = script_dir / ".env"

    print(f"Loading .env from directory: {env_path}")

    # Explicitly load from that specific path
    load_dotenv(dotenv_path=env_path)


def load_llm(by, model=None):
    if (by is not None):
        
        match by:
            case "HuggingFace":
                param = os.getenv("HF_MODEL"), os.getenv("HF_TOKEN"), os.getenv("CONTEXT_WINDOWS_SIZE")
                return LoadbyHuggingFace.load_llm(param)
            case "Ollama":
                param = os.getenv("OLLAMA_MODEL"), os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), os.getenv("CONTEXT_WINDOWS_SIZE")
                return LoadbyOllama.load_llm(param)
            case "LlamaCpp":
                param = os.getenv("LC_MODEL_PATH"), os.getenv("CONTEXT_WINDOWS_SIZE")
                return LoadbyLlamaCpp.load_llm(param)
            case "OpenRouter_API":
                param = os.getenv("OPENROUTER_API_KEY"), os.getenv("OPENROUTER_MODEL"), os.getenv("OPENROUTER_BASE_URL"), os.getenv("CONTEXT_WINDOWS_SIZE")
                return LoadbyOpenRouter.load_llm(param)
            case "Google_API":
                if (model is not None):
                    os.environ["GOOGLE_MODEL"] = model
                param = os.getenv("GOOGLE_API_KEY"), os.getenv("GOOGLE_MODEL"), os.getenv("CONTEXT_WINDOWS_SIZE")
                return LoadbyGoogle.load_llm(param)
            case "Nvidia_API":
                if (model is not None):
                    param = os.getenv("NVIDIA_API_KEY"), model, os.getenv("CONTEXT_WINDOWS_SIZE")
                else:
                    param = os.getenv("NVIDIA_API_KEY"), os.getenv("NVIDIA_MODEL"), os.getenv("CONTEXT_WINDOWS_SIZE")
                return LoadbyNvidia.load_llm(param)
            case _:
                return
    else:
        return None
    

def set_langsmith(project_name="My-Debug-Project"):
    LANGSMITH_KEY = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")

    os.environ["LANGSMITH_PROJECT"] = project_name
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_KEY
    os.environ["LANGSMITH_TRACING"] = LANGSMITH_TRACING

    if not os.environ.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter LangSmith API Key: ")
