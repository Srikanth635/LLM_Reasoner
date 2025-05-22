import operator
from typing import Annotated,Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END, MessagesState
from src.langchain.agents.framenet_agent import framenet_node, framenet_node_pal
from src.langchain.agents.flanagan_agent import flanagan_node, flanagan_node_pal
from src.langchain.agents.math_agent import math_node
from src.langchain.agents.websearch_agent import web_research_node, web_research_node_pal
from src.langchain.agents.pycram_agent import pycram_node_pal

from src.langchain.create_agents import *


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
# builder.add_node("director", director_node)
builder.add_node("web_researcher", web_research_node_pal)
builder.add_node("pycram", pycram_node_pal)
builder.add_node("framenet", framenet_node_pal)
builder.add_node("flanagan", flanagan_node_pal)
builder.add_node("aggregator", aggregator_node)
# builder.add_edge(START, "director")
builder.add_edge(START, "web_researcher")
builder.add_edge(START, "pycram")
builder.add_edge(START, "framenet")
builder.add_edge(START, "flanagan")
builder.add_edge("web_researcher", "aggregator")
builder.add_edge("pycram", "aggregator")
builder.add_edge("framenet", "aggregator")
builder.add_edge("flanagan", "aggregator")
builder.add_edge("aggregator", END)
pal_graph = builder.compile()


if __name__ == "__main__":
    print()






