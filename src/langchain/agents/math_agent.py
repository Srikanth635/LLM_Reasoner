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
def multiply(a: float, b: float):
    """Multiply two numbers."""
    print("INSIDE MULTIPLY TOOL")
    return a * b

@tool
def add(a: float, b: float):
    """Add two numbers."""
    print("INSIDE ADD TOOL")
    return a + b


# Agent
math_agent = create_agent(llm, [add,multiply])


# Agent Node
def math_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = math_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="mather")
            ]
        },
        goto="supervisor",
    )