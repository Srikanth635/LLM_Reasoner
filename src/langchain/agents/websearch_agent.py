from typing import Literal
from src.langchain.create_agents import *
from src.langchain.llm_configuration import *
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langgraph.types import Command

# Agent Specific Tools
web_search_tool = TavilySearchResults(max_results=2)


# Agent
websearch_agent = create_agent(llm, [web_search_tool])


# Agent Node
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