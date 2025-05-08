from typing import Literal
from src.langchain.create_agents import *
from src.langchain.llm_configuration import *
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langgraph.types import Command

# Agent Specific Tools
@tool
def framenet(a: str):
    """give the framenet representation of the input string"""
    print("INSIDE FRAMENET TOOL")
    return {'agent' : 'robot', 'patient' : 'apple'}


# Agent
framenet_agent = create_agent(llm, [framenet])


# Agent Node
def framenet_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = framenet_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="framenet")
            ]
        },
        goto="supervisor",
    )