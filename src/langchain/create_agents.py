import json
from typing import Annotated
from typing_extensions import TypedDict, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.messages.tool import ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from src.langchain.llm_configuration import *
from pydantic import Field

memory = MemorySaver()

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

def create_agent(llm, tools, agent_sys_prompt="", agent_state_schema: type = AgentState):
    print("Sub Agent Creation")
    # llm_ollama = ollama_llm.bind_tools(tools)
    llm_with_tools = llm.bind_tools(tools)

    if not agent_sys_prompt:
        agent_sys_prompt = SystemMessage(content="You are a smart agent and just pass on the tool output as it is with"
                                          "out any modification or further explanations")

    # Agent Node
    def chatbot(state: MessagesState):
        messages = [agent_sys_prompt] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    # Tool Node
    tool_node = BasicToolNode(tools=tools)

    graph_builder = StateGraph(agent_state_schema)
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

# --- Updated create_agent function ---
class FNBasicToolNode:
    def __init__(self, tools: list, agent_state_schema: type = AgentState):
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.agent_state = agent_state_schema

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No Message found in input")

        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            self.agent_state.framenet = tool_result
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

def create_framenet_agent(llm, tools, agent_sys_prompt="", agent_state_schema: type = AgentState):
    """
    Creates a framenet agent workflow graph.

    Args:
        llm: The language model to be used.
        tools: A list of tools the agent can use.
        agent_sys_prompt (str, optional): The system prompt for the agent.
                                         Defaults to a generic prompt.
        agent_state_schema (type, optional): The TypedDict class defining the agent's state.
                                             Defaults to AgentState.
    Returns:
        A compiled StateGraph representing the agent workflow.
    """

    llm_with_tools = llm.bind_tools(tools)

    if not agent_sys_prompt:
        # Ensure agent_sys_prompt is a SystemMessage if it's a string
        final_agent_sys_prompt = SystemMessage(
            content="You are a smart agent and just pass on the tool output as it is with"
                    "out any modification or further explanations"
        )
    elif isinstance(agent_sys_prompt, str):
        final_agent_sys_prompt = SystemMessage(content=agent_sys_prompt)
    else:
        final_agent_sys_prompt = agent_sys_prompt


    # Agent Node
    def chatbot(state: TypedDict): # Use dict here for more flexibility or the passed agent_state_schema
        # Ensure messages are handled correctly based on the agent_state_schema
        # This assumes 'messages' is a key in your state
        current_messages = state.get("messages", [])
        messages_for_llm = [final_agent_sys_prompt] + current_messages
        response = llm_with_tools.invoke(messages_for_llm)
        print("Framenet Agent Response:", response)
        return {"messages": [response]} # Ensure this matches how add_messages expects it


    # Tool Node
    tool_node = FNBasicToolNode(tools=tools, agent_state_schema=agent_state_schema)

    # Use the passed agent_state_schema for the graph
    graph_builder = StateGraph(agent_state_schema)
    graph_builder.add_node("agent", chatbot)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges(
        "agent",
        route_tools,
        {"tools": "tools", END: END}
    )
    graph_builder.add_edge("tools", "agent")
    graph_builder.set_entry_point("agent")
    return graph_builder.compile(checkpointer=True)


if __name__ == "__main__":
    print()
    # from src.langchain.agents.websearch_agent import web_search_tool
    #
    #
    # from dotenv import load_dotenv, find_dotenv
    #
    # load_dotenv(find_dotenv(), override=True)
    #
    # class TestState(TypedDict):
    #     user_query : str
    #     web_answer : str
    #
    # graph = create_agent2(ollama_llm, tools=[web_search_tool], agent_state_schema=TestState)
    #
    # config = {"configurable": {"thread_id": "1"}}
    #
    # print(graph.invoke({"user_query" : HumanMessage(content="who is 2023 ipl champions")}, config=config))
    #
    # print("-" *10)
    #
    # print(graph.get_state(config=config))

