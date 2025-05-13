from langgraph.graph import StateGraph, MessagesState, START

from src.langchain.agent_supervisor import supervisor_node
from src.langchain.agents.framenet_agent import framenet_node
from src.langchain.agents.math_agent import math_node
from src.langchain.agents.websearch_agent import web_research_node
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

builder = StateGraph(MessagesState)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("mather", math_node)
builder.add_node("web_researcher", web_research_node)
builder.add_node("framenet", framenet_node)
graph = builder.compile()


if __name__ == "__main__":
    print()

    # from IPython.display import display, Image
    # display(Image(graph.get_graph().draw_mermaid_png()))

    # Example: Complex Query Using Multiple Agents
    # input_question = "Find the founder of FutureSmart AI and then do a web research on him"
    # for s in graph.stream(
    #         {"messages": [("user", input_question)]},
    #         subgraphs=True
    # ):
    #     print(s)
    #     print("----")
