from src.langchain_flow.create_agents import *
from src.langchain_flow.agents.enhanced_ad_agent import enhanced_ad_agent_node
from src.langchain_flow.models_graph import models_node
from src.langchain_flow.state_graph import StateModel
from langgraph.graph import StateGraph,START

e2e_memory = MemorySaver()

# Aggregator Node
def aggregator_node(state: MessagesState):
    # messages = [
    #                {"role": "system", "content": framenet_system_prompt},
    #            ] + state["messages"]
    # print("Aggregator Node Messages", state["messages"])
    return {"messages" : "Message from Aggregator Node"}

def director_node(state: MessagesState):
    # messages = [
    #                {"role": "system", "content": framenet_system_prompt},
    #            ] + state["messages"]

    # director_agent = create_agent(ollama_llm,[])
    print("Director Node Messages", state["messages"])
    return state


builder = StateGraph(StateModel)
builder.add_node("ad_agent_node", enhanced_ad_agent_node)
builder.add_node("models_node", models_node)
builder.add_node("aggregator", aggregator_node)

builder.add_edge(START, "ad_agent_node")
builder.add_edge("ad_agent_node","models_node")
builder.add_edge("models_node", "aggregator")
builder.add_edge("aggregator", END)
e2e_graph = builder.compile()



if __name__ == "__main__":
    print()





