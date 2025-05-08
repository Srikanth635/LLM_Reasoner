from langchain_openai import ChatOpenAI

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.3
llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)