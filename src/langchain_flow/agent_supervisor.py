from typing_extensions import TypedDict
from typing import Literal
from langgraph.graph import MessagesState, END
from langgraph.types import Command

from src.langchain_flow.llm_configuration import *

#   For every new agent defined, it should be mentioned members, Router["next"] and supervisor_node() output format

# Define available agents
members = ["mather", "web_researcher", "framenet", "flanagan"]

# Add FINISH as an option for task completion
options = members + ["FINISH"]

# Create system prompt for supervisor
# system_prompt = (
#     "You are a supervisor tasked with managing a conversation between the"
#     f" following workers: {members}. Given the following user request,"
#     " respond with the worker to act next. Each worker will perform a"
#     " task and respond with their results and status. When finished,respond with FINISH."
# )
system_prompt = (
    f"You are a supervisor managing a conversation between the following workers: {members}. "
    "Given the user's request and previous responses, your job is to select the best next agent "
    "to continue solving the task — or respond with FINISH if the task is complete.\n\n"
    "Instructions:\n"
    "- Consider whether the user question has already been fully answered.\n"
    "- Choose ONLY the agent who is most relevant to the remaining part of the task.\n"
    "- Avoid calling unnecessary agents.\n"
    "- If the last response fully addresses the user request, respond with FINISH.\n\n"
    "Agents:\n"
    "- mather: performs basic mathematical calculations like addition, multiplication\n"
    "- web_researcher: looks up web/internet for current or factual information.\n"
    "- framenet: give a framenet representation (linguistic or semantic frame structures) of the robot task instruction\n"
    "- flanagan: provides and reasons about flanagan model representation of the given input NL robot action instruction\n"
)


# Define router type for structured output
class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["mather", "web_researcher", "framenet", "flanagan" , "FINISH"]

# Create supervisor node function
def supervisor_node(state: MessagesState) -> Command[Literal["mather", "web_researcher", "framenet", "flanagan", "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = ollama_llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    print(f"Next Worker: {goto}")
    if goto == "FINISH":
        goto = END
    return Command(goto=goto)


