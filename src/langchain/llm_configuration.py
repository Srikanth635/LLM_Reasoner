from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.3
llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama

ollama_llm = ChatOllama(model="qwen3:8b")