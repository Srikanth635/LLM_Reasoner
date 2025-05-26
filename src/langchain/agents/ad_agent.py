from langgraph.graph import MessagesState
from src.langchain.create_agents import *

agent_sys_prompt = ""


# ad_node = create_agent(llm=ollama_llm, tools=[], agent_sys_prompt=agent_sys_prompt)

def framenet_node_pal(state: MessagesState):

    agent_input_messages = [agent_sys_prompt] + state["messages"]

    result = ollama_llm.invoke(agent_input_messages)
    return {
        "messages": result["messages"][-1]
    }