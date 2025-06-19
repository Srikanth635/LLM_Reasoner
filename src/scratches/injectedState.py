from typing import List
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.tools import tool

from langgraph.prebuilt import InjectedState, ToolNode, create_react_agent
from src.langchain_flow.llm_configuration import *
from langgraph.graph import StateGraph, END, MessagesState

class AgentState(TypedDict):
    messages: List[BaseMessage]
    foo: str

@tool
def state_tool(x: int, state: Annotated[dict, InjectedState]) -> str:
    '''Do something with state.'''
    if len(state["messages"]) > 2:
        return state["foo"] + str(x)
    else:
        return "not enough messages"

@tool
def foo_tool(x: int, foo: Annotated[str, InjectedState("foo")]) -> str:
    '''Do something else with state.'''
    return foo + str(x + 1)

node = ToolNode([state_tool, foo_tool])

# tool_call1 = {"name": "state_tool", "args": {"x": 1}, "id": "1", "type": "tool_call"}
# tool_call2 = {"name": "foo_tool", "args": {"x": 1}, "id": "2", "type": "tool_call"}
# state = {
#     "messages": [AIMessage("", tool_calls=[tool_call1, tool_call2])],
#     "foo": "bar",
# }
# node.invoke(state)

graph = create_react_agent(
    model=ollama_llm,
    tools=[state_tool, foo_tool],
    prompt="You are a helpful assistant")

