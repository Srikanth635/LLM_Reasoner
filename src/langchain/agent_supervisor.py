from typing_extensions import TypedDict
from typing import Literal
from langgraph.graph import MessagesState, END
from langgraph.types import Command

from src.langchain.llm_configuration import *

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