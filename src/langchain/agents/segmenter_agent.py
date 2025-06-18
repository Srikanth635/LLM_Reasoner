# Input: [Segmented Video Descriptions]
#   ↓
# 1. Task Understanding Agent
#   ↓
# 2. Activity Segmentation Agent
#   ↓
# 3. Activity-to-Action Mapping Agent
#   ↓
# 4. Action Validation Agent
#   ↓
# Output: [Sequence of Robot Action Instructions]

# 1. Task Objective Inference Agent
# Input: All segments
# Output: Natural-language summary of the task (e.g., "Prepare a snack using Cheez-It crackers in a bowl").
#
# Purpose:
#
# Understand intent
#
# Guide filtering and abstraction of actions


from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from typing import TypedDict,List, Optional, Union, Dict
from src.langchain.llm_configuration import *

def llm_deduplicate_activities(activities):
    prompt = (
        "You are an assistant that cleans up a list of robot actions.\n"
        "Remove redundant or repeated steps that describe the same intent.\n\n"
        f"Actions:\n{activities}\n\n"
        "Return a concise list of unique actions required to complete the task."
    )
    result = ollama_llm.invoke({"input": prompt})
    # Parse result as list of dicts if structured
    return result.content


# Graph state structure
class SystemState(TypedDict):
    segments: List[str]
    task_objective: Optional[str]
    structured_activities: List[Union[str, Dict]]
    filtered_activities: List[Union[str, Dict]]
    final_instructions: List[str]


def initial_state(segments: List[str]) -> SystemState:
    return {
        "segments": segments,
        "task_objective": None,
        "structured_activities": [],
        "filtered_activities": [],
        "final_instructions": []
    }


# Define Nodes
def task_objective_agent(state : SystemState):
    segments = state["segments"]
    # LLM prompt goes here
    response = ollama_llm.invoke({"input": "\n".join(segments)})
    state["task_objective"] = response.content
    return state

def activity_extraction_agent(state : SystemState):
    segments = state["segments"]
    objective = state["task_objective"]
    # LLM prompt per segment
    activities = []
    for i, seg in enumerate(segments):
        prompt = f"Task: {objective}\nSegment {i+1}: {seg}\nWhat action is performed?"
        result = ollama_llm.invoke({"input": prompt})
        activities.append(result.content)  # Parse into structured dict
    state["structured_activities"] = activities
    return state

def redundancy_filter_agent(state : SystemState):
    acts = state["structured_activities"]
    # Heuristic or LLM-based deduplication
    deduped = llm_deduplicate_activities(acts)
    state["filtered_activities"] = deduped
    return state

def instruction_generator_agent(state : SystemState):
    filtered = state["filtered_activities"]
    # Map structured actions to natural language
    instructions = []
    for act in filtered:
        prompt = f"Convert this to an instruction: {act}"
        result = ollama_llm.invoke({"input": prompt})
        instructions.append(result.content)
    state["final_instructions"] = instructions
    return state

# LangGraph DAG construction
graph = StateGraph(SystemState)

graph.add_node("TaskObjectiveAgent", task_objective_agent)
graph.add_node("ActivityExtractionAgent", activity_extraction_agent)
graph.add_node("RedundancyFilterAgent", redundancy_filter_agent)
graph.add_node("InstructionGeneratorAgent", instruction_generator_agent)

graph.set_entry_point("TaskObjectiveAgent")
graph.add_edge("TaskObjectiveAgent", "ActivityExtractionAgent")
graph.add_edge("ActivityExtractionAgent", "RedundancyFilterAgent")
graph.add_edge("RedundancyFilterAgent", "InstructionGeneratorAgent")
graph.add_edge("InstructionGeneratorAgent", END)

# Compile
workflow = graph.compile()


if __name__ == "__main__":
    print()
    segments = [
        "The person grabs a Cheez-It box from the shelf.",
        "The person continues to hold the box.",
        "The person pours Cheez-Its into a red bowl.",
        "The person finishes pouring and sets the box down."
    ]

    state = initial_state(segments)
    result = workflow.invoke(state)
    print("\n".join(result["final_instructions"]))