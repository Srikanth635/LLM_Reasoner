from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.3
# llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

OLLAMA_MODEL = "qwen3:8b"

# from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama

try:
    print("Trying OLLAMA model 14b")
    ollama_llm = ChatOllama(model="qwen3:14b")
    ollama_llm_small = ChatOllama(model="qwen3:8b")
    print("OLLAMA model found")
except:
    print("OLLAMA model not found")
    ollama_llm = ChatOllama(model=OLLAMA_MODEL)
    ollama_llm_small = ChatOllama(model="qwen3:8b")

# ollama_llm = ChatOllama(model="qwen3:4b")

# ollama_llm = ChatOllama(model="gemma3:4b")