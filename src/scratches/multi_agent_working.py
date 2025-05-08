from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, ChatMessage, FunctionMessage
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from typing import Literal
from typing_extensions import TypedDict,Annotated,Sequence
from langgraph.graph.message import add_messages
from langchain_core.tools import Tool, tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState, START, END
from langgraph.types import Command

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


#Reference https://blog.futuresmart.ai/multi-agent-system-with-langgraph

# 1. Configuration
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.3
llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

def create_agent(llm, tools):
    llm_with_tools = llm.bind_tools(tools)
    def chatbot(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", chatbot)

    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
    )
    graph_builder.add_edge("tools", "agent")
    graph_builder.set_entry_point("agent")
    return graph_builder.compile()

# Tools
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

@tool
def framenet(a: str):
    """give the framenet representation of the input string"""
    print("INSIDE FRAMENET TOOL")
    return {'agent' : 'robot', 'patient' : 'apple'}


from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(max_results=2)


# Web Search Agent
websearch_agent = create_agent(llm, [web_search_tool])

def web_research_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = websearch_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="web_researcher")
            ]
        },
        goto="supervisor",
    )


# Addition Agent
math_agent = create_agent(llm, [add,multiply])
def addition_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = math_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="mather")
            ]
        },
        goto="supervisor",
    )

# Multiplication Agent
# multiplication_agent = create_agent(llm, [multiply])
# def multiplication_node(state: MessagesState) -> Command[Literal["supervisor"]]:
#     result = multiplication_agent.invoke(state)
#     return Command(
#         update={
#             "messages": [
#                 HumanMessage(content=result["messages"][-1].content, name="multiplier")
#             ]
#         },
#         goto="supervisor",
#     )

# Framenet Agent
framenet_agent = create_agent(llm, [framenet])
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

## Supervisor

# Define available agents
members = ["mather", "web_researcher", "framenet"]

# Add FINISH as an option for task completion
options = members + ["FINISH"]

# Create system prompt for supervisor
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

# Define router type for structured output
class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["mather", "web_researcher", "framenet", "FINISH"]

# Create supervisor node function
def supervisor_node(state: MessagesState) -> Command[Literal["mather", "web_researcher", "framenet", "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    print(f"Next Worker: {goto}")
    if goto == "FINISH":
        goto = END
    return Command(goto=goto)

builder = StateGraph(MessagesState)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("mather", addition_node)
builder.add_node("web_researcher", web_research_node)
builder.add_node("framenet", framenet_node)
graph = builder.compile()
