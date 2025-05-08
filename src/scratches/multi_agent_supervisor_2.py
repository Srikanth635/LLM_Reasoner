from __future__ import annotations
import json
import inspect
from typing import Annotated, List, Dict, Any, Optional, Type, Callable
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages.tool import tool_call, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool, tool
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv, find_dotenv
import owlready2
from owlready2 import *
from pathlib import Path
import types
load_dotenv(find_dotenv(), override=True)

# from ontology_singleton import *
# ONTOLOGY_FILE = "http://www.ease-crc.org/ont/SOMA.owl"
# om = OntologyManager(ONTOLOGY_FILE)

# 1. Configuration
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.3

# 2. State Definition
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 3. Tools (Non-static methods from OntologyManager)
@tool
def add(a: float, b: float):
    """Add two numbers."""
    return a + b

tools = [
    add
]

# 4. LLM and Tool Binding
llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
llm_with_tools = llm.bind_tools(tools)


# 5. Agent Node
def add_agent(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# 6. Tool Node
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


tool_node = BasicToolNode(tools=tools)


# 7. Routing Function
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


# 8. Graph Construction
graph_builder = StateGraph(State)
graph_builder.add_node("agent", add_agent)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("agent", route_tools, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")
graph_builder.add_edge(START, "agent")
graph2 = graph_builder.compile()


# 9. Stream Function
def stream_graph_updates2(user_input: str):
    state = {"messages": [HumanMessage(content=user_input)]}
    output = []
    for event in graph2.stream(state):
        for node_output in event.values():
            if "messages" in node_output:
                messages = node_output["messages"]
                if messages:
                    last_message = messages[-1]
                    if isinstance(last_message, HumanMessage):
                        output.append(f"Human Message: {last_message.content}")
                        print(f"Human Message: {last_message.content}")
                    elif isinstance(last_message, AIMessage):
                        output.append(f"AI Message: {last_message.content}")
                        print(f"AI Message: {last_message.content}")
                        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                            for tool_call in last_message.tool_calls:
                                output.append(f"  Tool: {tool_call['name']}")
                                output.append(f"  Tool Args: {tool_call['args']}")
                                print(f"  Tool: {tool_call['name']}")
                                print(f"  Tool Args: {tool_call['args']}")
                                tool_results = tool_node({"messages": [last_message]})
                                tool_result_message = tool_results["messages"][-1]
                                if isinstance(tool_result_message, ToolMessage):
                                    print(f"  Tool Message: {tool_result_message.content}")
                                    output.append(f"  Tool Message: {tool_result_message.content}")
                    elif isinstance(last_message, ToolMessage):
                        output.append(f"Tool Message: {last_message.content}")
                        print(f"Tool Message: {last_message.content}")
    return str(output)

#10. Direct Graph Invocation
def graph_invoke2(user_input: str):
    return graph2.invoke({"messages" : [HumanMessage(content=user_input)]})


from langgraph.prebuilt import create_react_agent
addition_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=tools,
    prompt=(
            "You are a addition agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with math-related addition tasks\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        ),
    name = "addition_agent"
)
# # 10. Example Usage
# if __name__ == "__main__":
#     # inputs = input("Enter your query: ")
#     # stream_graph_updates(inputs)
#     print()