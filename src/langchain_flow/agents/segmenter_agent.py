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
from src.langchain_flow.llm_configuration import *
import re
import json
from pydantic import BaseModel,Field

class StructuredActivity(BaseModel):
    intent: str
    object: Optional[str] = None
    target: Optional[str] = None
    tool: Optional[str] = None
    location: Optional[str] = None
    modifier: Optional[str] = None
    quantity: Optional[str] = None
    duration: Optional[str] = None

class OrderedInstructions(BaseModel):
    instructions: List[str] = Field(description="List of Robot Actionable sequence of instructions in Natural language")


TaskObjectiveAgent_prompt_template = """
    You are an intelligent task reasoning assistant.

    You are given a list of sequential descriptions from a segmented video, each describing a person's actions in a scene. Your job is to
    infer the high-level task or objective the person is trying to complete across the entire sequence.
    
    Consider the overall progression of objects used, goals achieved, and final outcomes. Think beyond superficial actions and
    identify the deeper intent of the task.
    
    Respond with one concise sentence that clearly expresses the overall objective of the task.
    
    ---
    
    /nothink
"""

ActivityExtractionAgent_prompt_template = """
    You are an assistant that extracts structured activity information from natural language video segment descriptions.

    You will be given:
    - A high-level task objective
    - A single segment of description
    
    Your job is to identify the **atomic action(s)** performed in this segment, as well as all related **entities** and **roles**. If multiple
    distinct actions are performed in one segment, output each one as a separate entry.
    
    ** IMPORTANT CONSIDERATION **
    You must restrict the "intent" field to one of the following allowed robot action classes:

    [peel, cut, pick up, lift, open, operate tap, pipette, pour, press, pull, place, remove, roll, shake, spoon, sprinkle, stir, take, turn, unscrew, wait]
    
    Only use these verbs in the "intent" field. Do not invent new actions. If an action doesn't fit, select the closest appropriate one.
    
    Return the result as a JSON array of structured activity objects. Use the format:
    
    ```json
    [
      {
        "intent": "<core action verb, e.g., pick up, place, pour, open>",
        "object": "<main object being acted on>",
        "target": "<optional target or destination>",
        "tool": "<optional tool used, if any>",
        "modifier": "<optional phrase like slightly, carefully, if specified>",
        "location": "<optional location reference, e.g., counter, cupboard>",
        "quantity": "<optional quantity e.g., 2 slices, 200 ml, a spoonful>",
        "duration": "<optional duration e.g., 10 seconds, 5 minutes>"
      }
    ]
    
    Include all objects, tools, modifiers, or locations mentioned. If a field is not applicable, omit it.

    Be accurate, extract only what is clearly described, and do not make assumptions.
    Respond with a valid JSON array of structured actions .
    
    ---

"""

RedundancyFilterAgent_prompt_template = """
    You are a function that receives a JSON array of structured robot actions and returns a cleaned version of that list.
    
    Rules:
    - Remove duplicate or overlapping actions.
    - Merge fine-grained variations of the same action (e.g., "adjust", "realign").
    - Do NOT invent or describe actions.
    - Do NOT include commentary, explanation, notes, formatting, or headings.
    - Do NOT return anything except the JSON array.
    
    Input: A JSON array of action objects.
    
    Return only: A simplified JSON array of unique actions.
    
    /nothink
"""

InstructionGeneratorAgent_prompt_template = """
    You are a robot instruction generator for a physically realistic one-armed robot assistant.
    
    Given a structured activity describing an action, convert it into a clear and precise natural-language command that a robot or agent could execute.
    
    You must follow these guidelines:
    
    1. The robot has only **one working arm**.
    2. If the robot **picks up** or **holds** an object, it must **place** or **release** that object before picking up another.
    3. Do not generate commands that involve holding multiple objects at the same time.
    4. If the robot needs to switch from one object to another, insert an intermediate instruction (e.g., “Place the current object on the counter”) **before** the next action.
    5. Track whether the robot is currently holding an object (you will be informed of this, or you can infer it based on recent instructions).
    6. Maintain physical realism and logical object handoff.
    
    Include important details in your instructions:
    - Action (intent)
    - Object involved
    - Target or location (if present)
    - Tool used (if any)
    - Modifier (e.g., carefully, slightly), if relevant
    
    Only generate instructions using the following allowed robot actions:
    
    [peel, cut, pick up, lift, open, operate tap, pipette, pour, press, pull, place, remove, roll, shake, spoon, sprinkle, stir, take, turn, unscrew, wait]
    
    **Do not invent new actions or verbs.** Match each instruction to its corresponding allowed action type.
    
    Use polite and fluent imperative form.
    
    Examples:
    ```json
    { "intent": "pick up", "object": "knife", "location": "counter" }
    → "Pick up the knife from the counter."
    
    { "intent": "pour", "object": "water bottle", "target": "glass", "tool": "water bottle" }
    → "Pour the water from the bottle into the glass."
    
    { "intent": "pick up", "object": "cloth" } followed by { "intent": "pick up", "object": "jar" } while still holding the cloth
    → "Place the cloth on the counter.\nPick up the jar."
    
    Return only the final instruction(s) in natural language, inserting release/place commands if needed for physical feasibility.
    Do not add any commentary, feedback or explanation to it. The output should be just the instruction.

    ---
    
    /nothink
"""

InstructionRefinementAgent_prompt_template = """
You are a task optimizer and robot reasoning assistant.

You are given a list of robot instructions written in natural language. These commands describe physical actions for a robot with one arm to perform.

Your task is to:
- Reorder the steps if needed to match a logical and physically feasible sequence.
- Remove redundant, duplicate, or conflicting actions.
- Ensure that object manipulations respect physical constraints (e.g., the robot must release an object before picking up another).
- Improve command phrasing for clarity, precision, and direct robot execution.
- Only use action phrases from this allowed action set:

[peel, cut, pick up, lift, open, operate tap, pipette, pour, press, pull, place, remove, roll, shake, spoon, sprinkle, stir, take, turn, unscrew, wait]

### EXPECTED OUTPUT: ###
** Return only the final, corrected instruction list — one instruction per line in execution order.
Do not add commentary, explanation, or formatting or feedback. **

/nothink
"""

def think_remover(res : str):
    if re.search(r"<think>.*?</think>", res, flags=re.DOTALL):
        cleaned_res = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip()
    else:
        cleaned_res = res.strip()

    return cleaned_res

def llm_deduplicate_activities(activities):
    prompt = (
            RedundancyFilterAgent_prompt_template + "\n" +
            f"Actions:\n{activities}\n\n"
            "Return a concise list of unique actions required to complete the task."
    )
    result = ollama_llm.invoke(prompt)
    try:
        return json.loads(think_remover(result.content))
    except json.JSONDecodeError:
        return [think_remover(result.content)]

# def llm_deduplicate_activities(activities):
#     prompt = (
#         RedundancyFilterAgent_prompt_template + "\n" +
#         f"Actions:\n{activities}\n\n"
#         "Return a concise list of unique actions required to complete the task."
#     )
#     result = ollama_llm.invoke(prompt)
#     # Parse result as list of dicts if structured
#     return think_remover(result.content)

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
    segments_str = "\n".join(segments)
    # LLM prompt goes here
    prompt  = TaskObjectiveAgent_prompt_template + "\n" + f"segmented_descriptions: {segments_str}"
    response = ollama_llm.invoke(prompt)
    state["task_objective"] = think_remover(response.content)
    print("Task Objective: " + state["task_objective"] + "\n")
    return state

def activity_extraction_agent(state : SystemState):
    segments = state["segments"]
    objective = state["task_objective"]
    # LLM prompt per segment
    activities = []
    for i, seg in enumerate(segments):
        prompt = ActivityExtractionAgent_prompt_template + "\n" + f"Task: {objective}\nSegment {i+1}: {seg}\nWhat action is performed?"
        ollama_llm_structured = ollama_llm.with_structured_output(StructuredActivity, method="json_schema")
        result = ollama_llm_structured.invoke(prompt)
        activities.append(result.model_dump())  # Parse into structured dict
    state["structured_activities"] = activities
    print("Structured Activities: " + str(activities) + "\n")
    return state

def redundancy_filter_agent(state : SystemState):
    acts = state["structured_activities"]
    # Heuristic or LLM-based deduplication
    deduped = llm_deduplicate_activities(acts)
    state["filtered_activities"] = deduped
    print("Filtered Activities: " + str(deduped) + "\n")
    return state

import json

def instruction_generator_agent(state : SystemState):
    filtered = state["structured_activities"]
    # Map structured actions to natural language
    instructions = []
    for act in filtered:
        prompt = InstructionGeneratorAgent_prompt_template + "\n" +f"Convert this to an instruction: {act}"
        result = ollama_llm.invoke(prompt)
        instructions.append(think_remover(result.content))
        print("Converted instruction : ",think_remover(result.content))
    state["final_instructions"] = instructions
    print("Final Instructions: " + str(instructions) + "\n")
    return state

def instruction_refinement_agent(state: SystemState):
    raw_instructions = state["final_instructions"]
    joined = "\n".join(raw_instructions)

    prompt = InstructionRefinementAgent_prompt_template.strip() + "\n\nInstructions:\n" + joined
    ollama_llm_structured = ollama_llm.with_structured_output(OrderedInstructions, method="json_schema")
    result = ollama_llm_structured.invoke(prompt)
    # result_cleaned = think_remover(result.content)
    # refined_lines = [
    #     line.strip()
    #     for line in result.content.strip().splitlines()
    #     if line.strip() and not line.strip().startswith(("Final answer", "Instructions:", "---"))
    # ]
    state["final_instructions"] = result.model_dump()["instructions"]
    print("Refined Instructions: ", state["final_instructions"])
    # state["final_instructions"] = refined_lines
    # print("Refined Instructions:\n", "\n".join(refined_lines))
    return state

# LangGraph DAG construction
graph = StateGraph(SystemState)

graph.add_node("TaskObjectiveAgent", task_objective_agent)
graph.add_node("ActivityExtractionAgent", activity_extraction_agent)
graph.add_node("RedundancyFilterAgent", redundancy_filter_agent)
graph.add_node("InstructionGeneratorAgent", instruction_generator_agent)
graph.add_node("InstructionRefinementAgent", instruction_refinement_agent)


graph.set_entry_point("TaskObjectiveAgent")
graph.add_edge("TaskObjectiveAgent", "ActivityExtractionAgent")
graph.add_edge("ActivityExtractionAgent", "RedundancyFilterAgent")
graph.add_edge("RedundancyFilterAgent", "InstructionGeneratorAgent")
graph.add_edge("InstructionGeneratorAgent", "InstructionRefinementAgent")
graph.add_edge("InstructionRefinementAgent", END)

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

    complex_segments = [
        "In Segment 1, the person walks to the kitchen counter where a loaf of bread, a knife, a jar of peanut butter, and a plate are placed.",
        "In Segment 2, the person opens the loaf of bread and pulls out two slices, placing them on the plate.",
        "In Segment 3, the person adjusts the slices of bread slightly to align them better on the plate.",
        "In Segment 4, the person picks up the knife and unscrews the lid of the peanut butter jar.",
        "In Segment 5, the person dips the knife into the peanut butter jar to scoop out some spread.",
        "In Segment 6, the person spreads peanut butter on one of the slices of bread.",
        "In Segment 7, the person scrapes a bit more peanut butter and continues spreading on the other slice.",
        "In Segment 8, the person places one slice of bread on top of the other, completing the sandwich.",
        "In Segment 9, the person adjusts the top slice of the sandwich slightly to align it properly.",
        "In Segment 10, the person picks up the plate with the sandwich.",
        "In Segment 11, the person carries the plate to the sink area.",
        "In Segment 12, the person places the plate with the sandwich on the counter next to the sink.",
        "In Segment 13, the person wipes the counter with a cloth.",
        "In Segment 14, the person picks up the peanut butter jar and screws the lid back on.",
        "In Segment 15, the person puts the peanut butter jar back into the cupboard.",
        "In Segment 16, the person rinses the knife under the sink faucet."
    ]

    my_segments = [
        "the person begins preparing for a task by gathering a box of Cheez-It crackers, a red bowl, a red cup, and a bottle of mustard, likely to create a snack or dish.",
        "the person approaches the box of Cheez-It crackers to open it, signaling the start of the pouring task.",
        "the person grasps the box of Cheez-It crackers with both hands, holding it steady on the table as part of the grasping phase.",
        "the person begins grasping-and-moving the box of Cheez-It crackers, pouring them into the red bowl while the red cup and mustard are ready for use.",
        "the person continues pouring Cheez-It crackers into the red bowl, maintaining the grasp-and-move phase as they prepare the snack.",
        "the person refines the pouring motion, ensuring the crackers are evenly distributed into the bowl while the other items remain in place.",
        "the person adjusts the pouring technique, carefully transferring the crackers into the bowl as part of the grasp-and-move phase.",
        "the person completes the pouring action, moving the box away from the bowl as the grasp-and-move phase concludes.",
        "the person prepares to release the box of Cheez-It crackers, signaling the final phase of the task.",
        "the person positions the box for release, ensuring the crackers are fully poured into the bowl before completing the task.",
        "the person releases the box of Cheez-It crackers, finishing the pouring task and leaving the red bowl, cup, and mustard ready for the final snack preparation."
    ]

    state = initial_state(my_segments)
    result = workflow.invoke(state)
    print("\n".join(result["final_instructions"]))