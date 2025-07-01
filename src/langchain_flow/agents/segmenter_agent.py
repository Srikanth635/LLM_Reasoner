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
    instructions: List[str] = Field(description="List of Robot Actionable instructions in natural language imperative form.")

counter = 0

class Reflection(BaseModel):
    reflections : List[str] = Field(description="List of feedbacks/recommendations/reasons sentences")
    # need_correction : bool = Field(default=False, description="corrections to the instructions are needed or not.")

TaskObjectiveAgent_prompt_template = """
    You are an intelligent task reasoning assistant.

    You are given a list of sequential descriptions from a segmented video, each describing a person's actions in a scene. Your job is to
    infer the high-level task or objective the person is trying to complete across the entire sequence.
    
    To do this, first consider the sequence of actions as a whole. Pay attention to the initial state of the objects, how they are transformed
    or combined, and the final state of the scene. Think beyond the individual actions to identify the deeper intent.
    
    Respond with one concise sentence that clearly expresses the overall objective of the task.
    
    ---
    
    /nothink
"""

ActivityExtractionAgent_prompt_template = """
    You are a precise and methodical assistant that extracts structured activity information from natural language. Your goal is to be accurate and concise.

    You will be given:
    - A high-level task objective
    - A single segment of description

    Your job is to identify the **atomic action(s)** performed in this segment, as well as all related **entities** and **roles**. Use the
    provided **high-level task objective** to help disambiguate the intent of an action if the description is vague. If multiple distinct
    actions are performed in one segment, output each one as a separate entry.

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
    Respond with a valid JSON array of structured actions.

    ---
"""

RedundancyFilterAgent_prompt_template = """
    You are a function that receives a JSON array of structured robot actions and returns a cleaned version of that list.
    
    Rules:
    - Remove actions that are functionally identical and consecutive. An action is a duplicate if it has the same "intent" and "object"
        as the one immediately preceding it. If two consecutive actions are identical but one contains more detail (e.g., a "location" field),
        keep only the more detailed one.
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
    4. **Crucial Rule:** Before generating an instruction for a "pick up" or "take" action, you must check the preceding context. If the robot is
        already holding an item, you **MUST** first generate a command to "Place" the current item on a stable surface (e.g., "Place the bottle on the counter.")
        before generating the new "pick up" command.
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

StatefulInstructionPlannerAgent_prompt_template = """
    You are an expert robot motion planner for a one-armed robot assistant.

    You will be given a JSON array of sequential actions extracted from a video. Your task is to convert this entire sequence into a physically realistic, efficient, and
    logical list of natural language instructions.

    You must follow these rules meticulously:

    1.  **Process the ENTIRE list of actions**: Do not process each action in isolation. Consider the full sequence to understand the context.
    2.  **State Tracking (One-Arm Constraint)**: The robot has only one arm.
        -   Maintain an internal state of what the robot is currently holding. Let's call this `held_item`. Initially, `held_item` is `null`.
        -   When you encounter a "pick up" or "take" action, set `held_item` to that object.
        -   If you encounter a "pick up" or "take" action when `held_item` is NOT `null`, you **MUST** first generate a "Place [current held_item] on the counter" instruction
            before generating the new "pick up" instruction.
        -   After a "place", "pour", "sprinkle", or "spoon" action where the tool is released, update `held_item` to `null`.
    3.  **Redundancy Elimination**:
        -   Examine the incoming actions for redundancy. If you see multiple, slightly different actions that achieve the same goal (e.g., "lift cup", "pick up cup"), merge them
            into a single, clear instruction.
        -   Remove any action that is a minor, unnecessary part of another action (e.g., ignore "move hand towards knife" if it's followed by "pick up knife").
    4.  **Instruction Generation**:
        -   Generate one clear, concise instruction per line from the allowed action list.
        -   Use a polite, imperative tone (e.g., "Pick up the knife," "Pour the water...").
        -   Only use verbs from this list for the core action: `[peel, cut, pick up, lift, open, operate tap, pipette, pour, press, pull, place, remove, roll, shake, spoon,
            sprinkle, stir, take, turn, unscrew, wait]`

    **Example Walkthrough:**
    Input JSON array:
    ```json
    [
      {{ "intent": "lift", "object": "bottle" }},
      {{ "intent": "pick up", "object": "bottle" }},
      {{ "intent": "unscrew", "object": "cap", "tool": "bottle" }},
      {{ "intent": "pick up", "object": "glass" }}
    ]
    ```

    **Your internal thought process should be:**
    1.  `held_item` is `null`.
    2.  Action 1 ("lift bottle") and 2 ("pick up bottle") are redundant. I will merge them into one: "Pick up the bottle."
    3.  After this, `held_item` is "bottle".
    4.  Action 3 ("unscrew cap") uses the bottle. The robot is holding it. This is valid. Instruction: "Unscrew the cap from the bottle." `held_item` is still "bottle".
    5.  Action 4 is "pick up glass". But `held_item` is "bottle". I MUST place the bottle first.
    6.  So, I will insert an instruction: "Place the bottle on the counter."
    7.  Now, `held_item` is `null`.
    8.  I can now process the "pick up glass" action. Instruction: "Pick up the glass."
    9.  Final `held_item` is "glass".

    **Final Output for Example:**
    Pick up the bottle.
    Unscrew the cap from the bottle.
    Place the bottle on the counter.
    Pick up the glass.
    
    
    NOTE: Return only the final instruction(s) in natural language. Do not add any commentary, feedback or explanation around them. The output should be just the instructions.
    ---
    Now, process the following JSON array and provide the final list of instructions. Return ONLY the instructions, one per line.

"""

InstructionRefinementAgent_prompt_template = """
    You are a meticulous quality assurance assistant for robot task plans.
    
    You will be given a pre-generated list of natural language instructions for a one-armed robot. The list should already be mostly correct and logically ordered.
    
    Your task is to perform a final review with these priorities:
    
    1.  **Strictly Verify Physical Constraints**: Read through the instructions one last time, ensuring there are no violations of the one-armed robot rule. For every "pick up", confirm that the robot's hand was free. This is your most important task.
    2.  **Check for Logical Flow**: Ensure the overall plan makes sense. For instance, an object should be opened before its contents are poured. Do not reorder steps unless there is a clear logical contradiction.
    3.  **Improve Clarity and Conciseness**: Rephrase any awkward or ambiguous commands. Ensure every command uses an action verb from the approved list: `[peel, cut, pick up, lift, open, operate tap, pipette, pour, press, pull, place, remove, roll, shake, spoon, sprinkle, stir, take, turn, unscrew, wait]`
    
    NOTE: Additionally, you might be given an optional feedback/reason/recommendation on your refinement instructions. If given consider them and alter the
    instructions accordingly only if needed.
    
    ### EXPECTED OUTPUT: ###
    ** Return only the final, corrected instruction list — one instruction per line in execution order.
    Do not add commentary, explanation, formatting, or feedback. **
"""

ReflectionAgent_prompt_template = """
    You are a meticulous quality review assistant for robot task plans.
    
    You're are provided with the overall objective and the sequence of atomic robot instructions to accomplish that.
    
    Perform logical checks on the sequence of robot action instructions. Provide critical feedback and recommendations about
    any error prone instructions or absurd order of action instructions or if there's any redundancy or repetition of instructions.
    
    Give out all the reasons/feedbacks/recommendations as a list of statements.
"""

InstructionRefinementAgent_prompt_template_old = """
You are a task optimizer and robot reasoning assistant.

You are given a list of robot instructions written in natural language. These commands describe physical actions for a robot with one arm to perform.

Your task is to:
- Review the sequence for physical and logical feasibility. **Your primary goal is to ensure physical feasibility. Do NOT reorder steps unless the original
    sequence is physically impossible or illogical** (e.g., a 'cut' command comes before 'pick up knife'). When no reordering is necessary, preserve the original sequence.
- Remove any remaining redundant or conflicting actions.
- Perform a final check to ensure all physical constraints are met (e.g., the robot must place an object before picking up another).
- Refine command phrasing for clarity and conciseness.
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
    counter: int
    segments: List[str]
    task_objective: Optional[str]
    structured_activities: List[Union[str, Dict]]
    filtered_activities: List[Union[str, Dict]]
    final_instructions: List[str]
    reflections : List[str]


def initial_state(segments: List[str]) -> SystemState:
    return {
        "counter": 0,
        "segments": segments,
        "task_objective": None,
        "structured_activities": [],
        "filtered_activities": [],
        "final_instructions": [],
        "reflections":[]
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

def instruction_planner(state : SystemState):
    activities = state["structured_activities"]
    prompt = (
            StatefulInstructionPlannerAgent_prompt_template + "\n" +
            f"Actions:\n{activities}\n\n"
            "Return a concise list of unique actions required to complete the task."
    )
    structured_llm = ollama_llm.with_structured_output(OrderedInstructions, method="json_schema")
    result = structured_llm.invoke(prompt)
    # cleaned_result = think_remover(result.content).split("\n")
    print("Planned instructions : ", result.instructions , type(result.instructions))
    state['final_instructions'] = result.instructions
    return state


def instruction_refinement_agent(state: SystemState):
    raw_instructions = state["final_instructions"]
    joined = "\n".join(raw_instructions)

    joined_reflections = ""
    feedback = state['reflections']
    if feedback:
        joined_reflections = "\n".join(feedback)

    prompt = InstructionRefinementAgent_prompt_template.strip() + "\n\nInstructions:\n" + joined + "\n\nFeedback:\n" + joined_reflections
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
    state['counter'] = state['counter'] + 1
    # state["final_instructions"] = refined_lines
    # print("Refined Instructions:\n", "\n".join(refined_lines))
    return state

def reflection_agent(state : SystemState):
    objective = state["task_objective"]
    refined_instructions = state["final_instructions"]
    joined = "\n".join(refined_instructions)

    prompt = ReflectionAgent_prompt_template.strip() + "\nOverall Task Objective:"+objective  + "\n\nInstructions:\n" + joined
    ollama_llm_structured = ollama_llm.with_structured_output(Reflection, method="json_schema")
    result = ollama_llm_structured.invoke(prompt)
    state['counter'] = state['counter'] + 1
    state['reflections'] = result.reflections
    print("Reflections: ", state["reflections"])
    return state


# LangGraph DAG construction
graph = StateGraph(SystemState)

graph.add_node("TaskObjectiveAgent", task_objective_agent)
graph.add_node("ActivityExtractionAgent", activity_extraction_agent)
# graph.add_node("RedundancyFilterAgent", redundancy_filter_agent)
# graph.add_node("InstructionGeneratorAgent", instruction_generator_agent)
graph.add_node("InstructionPlannerAgent", instruction_planner)
graph.add_node("InstructionRefinementAgent", instruction_refinement_agent)
graph.add_node("ReflectionAgent", reflection_agent)


graph.set_entry_point("TaskObjectiveAgent")
graph.add_edge("TaskObjectiveAgent", "ActivityExtractionAgent")
# graph.add_edge("ActivityExtractionAgent", "RedundancyFilterAgent")
# graph.add_edge("RedundancyFilterAgent", "InstructionGeneratorAgent")
graph.add_edge("ActivityExtractionAgent", "InstructionPlannerAgent")
graph.add_edge("InstructionPlannerAgent", "InstructionRefinementAgent")

def should_continue(state : SystemState):
    counter = state["counter"]
    if counter > 3:
        # End after 5 iterations
        counter += 1
        return END
    return "ReflectionAgent"

graph.add_conditional_edges("InstructionRefinementAgent", should_continue)
graph.add_edge("ReflectionAgent", "InstructionRefinementAgent")

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