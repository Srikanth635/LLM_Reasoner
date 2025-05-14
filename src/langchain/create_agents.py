import json
from typing import Annotated
from typing_extensions import TypedDict, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.messages.tool import ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.langchain.llm_configuration import *

class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[list, add_messages]
    # messages: Annotated[Sequence[BaseMessage], add_messages]

class BasicToolNode:
    def __init__(self, tools: list):
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No Message found in input")

        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

# Routing Function
def route_tools(state: dict):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

def create_agent(llm, tools):

    llm_ollama = ollama_llm.bind_tools(tools)
    llm_with_tools = llm.bind_tools(tools)

    # Agent Node
    def chatbot(state: AgentState):
        messages = [SystemMessage(content="You are a smart agent and just pass on the tool output as it is with"
                                          "out any modification or further explanations")] + state["messages"] + [HumanMessage(content=""
                                           "/no_think")] #Dont add any additional explanation
        return {"messages": [llm_ollama.invoke(messages)]}

    # Tool Node
    tool_node = BasicToolNode(tools=tools)

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", chatbot)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges(
        "agent",
        route_tools,
        {"tools": "tools", END:END}
    )
    graph_builder.add_edge("tools", "agent")
    graph_builder.set_entry_point("agent")
    return graph_builder.compile()