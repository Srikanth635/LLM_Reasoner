import operator
from typing import Annotated,Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START,END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from src.langchain.agents.framenet_agent import framenet_node, framenet_node_pal, framenet_node_pal_custom
from src.langchain.agents.ad_agent import ad_agent_node_pal
from src.langchain.agents.flanagan_agent import flanagan_node_pal, flanagan_premotion_node, flanagan_phaser_node, flanagan_repr
from src.langchain.agents.math_agent import math_node
from src.langchain.agents.websearch_agent import web_research_node, web_research_node_pal
from src.langchain.agents.pycram_agent import pycram_node_pal
from langgraph.prebuilt import create_react_agent
from src.langchain.create_agents import *
from src.langchain.agents.enhanced_ad_agent import *
from src.langchain.state_graph import StateModel

memory2 = MemorySaver()

# Aggregator Node
def aggregator_node(state: MessagesState):
    # messages = [
    #                {"role": "system", "content": framenet_system_prompt},
    #            ] + state["messages"]
    # print("Aggregator Node Messages", state["messages"])
    return {"messages" : "Message from Aggregator Node"}

model_builder = StateGraph(StateModel)

model_builder.add_node("framenet", framenet_node_pal_custom)
model_builder.add_node("flanagan_premotion_node",flanagan_premotion_node )
model_builder.add_node("flanagan_phaser_node", flanagan_phaser_node)
model_builder.add_node("flanagan_repr", flanagan_repr)
model_builder.add_node("aggregator", aggregator_node)

model_builder.add_edge(START, "framenet")
model_builder.add_edge(START, "flanagan_premotion_node")

model_builder.add_edge("flanagan_premotion_node", "flanagan_phaser_node")
model_builder.add_edge("flanagan_phaser_node", "flanagan_repr")
model_builder.add_edge("flanagan_repr", "aggregator")
model_builder.add_edge("framenet", "aggregator")
model_builder.add_edge("aggregator", END)

models_graph = model_builder.compile(checkpointer=memory2)