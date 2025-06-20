import operator
from typing import Annotated,Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from src.langchain.agents.framenet_agent import framenet_node, framenet_node_pal
from src.langchain.agents.ad_agent import ad_agent_node_pal
from src.langchain.agents.flanagan_agent import flanagan_node, flanagan_node_pal
from src.langchain.agents.math_agent import math_node
from src.langchain.agents.websearch_agent import web_research_node, web_research_node_pal
from src.langchain.agents.pycram_agent import pycram_node_pal
from langgraph.prebuilt import create_react_agent
from src.langchain.create_agents import *

memory = MemorySaver()
class SharedState(TypedDict):
    mather : str
    web_researcher : str
    framenet : str
    flanagan : str


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


builder = StateGraph(MessagesState)
builder.add_node("action_designator", ad_agent_node_pal)
builder.add_node("web_researcher", web_research_node_pal)
builder.add_node("pycram", pycram_node_pal)
builder.add_node("framenet", framenet_node_pal)
# builder.add_node("flanagan", flanagan_node_pal)
builder.add_node("aggregator", aggregator_node)

builder.add_edge(START, "action_designator")
builder.add_edge("action_designator", "web_researcher")
builder.add_edge("action_designator", "pycram")
builder.add_edge("action_designator", "framenet")
# builder.add_edge("action_designator", "flanagan")

builder.add_edge("web_researcher", "aggregator")
builder.add_edge("pycram", "aggregator")
builder.add_edge("framenet", "aggregator")
# builder.add_edge("flanagan", "aggregator")
builder.add_edge("aggregator", END)
pal_graph = builder.compile(checkpointer=memory)


# builder = StateGraph(MessagesState)
# builder.add_node("action_designator", ad_agent_node_pal)
# builder.add_node("aggregator", aggregator_node)
#
# builder.add_edge(START, "action_designator")
# builder.add_edge("action_designator", "aggregator")
#
# builder.add_edge("aggregator", END)
# pal_graph = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    print()
    config = {"configurable" : {"thread_id" : "1", "user_id" : "srikanth123"}}
    pal_graph.invoke({"messages" : [HumanMessage(content="pick the mug")]}, config=config)
    print("#"*10)
    print(pal_graph.get_state(config=config, subgraphs=True))





