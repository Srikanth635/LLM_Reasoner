from src.langchain_flow.create_agents import create_agent
from src.resources.pycram.pycram_action_designators import (MoveTorsoAction, SetGripperAction,
        GripAction, ParkArmsAction, NavigateAction, PickUpAction, PlaceAction, ReachToPickUpAction, TransportAction,
        LookAtAction, OpenAction, CloseAction, GraspingAction, MoveAndPickUpAction, MoveAndPlaceAction, FaceAtAction,
                                                            DetectAction, SearchAction)
from src.resources.pycram.pycram_action_designators import *
from langchain_core.tools import tool
from langchain_core.tools.structured import StructuredTool
from langchain.agents import Tool
from src.langchain_flow.create_agents import *
from src.langchain_flow.llm_configuration import *
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Type, List, Literal, Union, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from langgraph.types import Command
from enum import Enum

## Locals
answers = {
    "instruction" : "",
    "model_names" : "",
    "populated_models" : ""
}


model_selector_prompt_template = """
    You are an intelligent robotic action classifier.

    Your task is to read a user instruction and select the most relevant robot action model(s) from the available list. Each action model represents a specific robotic capability. Return the name(s) of the model(s) that best match the instruction.

    Only respond with a list of model names, exactly as shown in the list below. Do not return structured instances or JSON fields — only the model name(s) as strings.

    You may return:
    - A single model name (e.g., "PickUpAction")
    - A list of model names (e.g., ["NavigateAction", "LookAtAction"]) if multiple actions are implied
    - Nothing (`[]`) if no model is relevant

    ### Available Action Models:
    - PickUpAction - Let the robot pick up an object.
    - PlaceAction - Places an Object at a position using an arm.
    - NavigateAction - Navigates the Robot to a position.
    - SetGripperAction - Set the gripper state of the robot.
    - LookAtAction - Lets the robot look at a position.
    - MoveTorsoAction - Move the torso of the robot up and down.
    - GripAction - Grip an object with the robot.
    - ParkArmsAction - Park the arms of the robot.
    - MoveAndPickUpAction - Navigate to `standing_position`, then turn towards the object and pick it up.
    - MoveAndPlaceAction - Navigate to `standing_position`, then turn towards the object and pick it up.
    - OpenAction - Opens a container like object
    - CloseAction - Closes a container like object.
    - FaceAtAction - Turn the robot chassis such that is faces the ``pose`` and after that perform a look at action.
    - ReachToPickUpAction - Let the robot reach a specific pose.
    - DetectAction - Detects an object that fits the object description and returns an object designator_description 
                    describing the object. If no object is found, an PerceptionObjectNotFound error is raised.
    - TransportAction - Transports an object to a position using an arm
    - SearchAction - Searches for a target object around the given location.
    - GraspingAction - Grasps an object described by the given Object Designator description

    ### Examples:

    #### Instruction:
    "Pick up the red mug from the counter using the left arm."
    #### Output:
    ["PickUpAction"]

    #### Instruction:
    "Move to the table, face the object, and pick it up."
    #### Output:
    ["NavigateAction", "FaceAtAction", "PickUpAction"]

    #### Instruction:
    "Nothing needs to be done."
    #### Output:
    []

    Respond with only the list of model names.

    Now, for the given natural language instruction {input_instruction} generate output as list of model names
"""

model_populator_prompt_template = """
You are a structured reasoning assistant for a robotics system.

You are given a list of robot action model names that correspond to available robotic capabilities.
Your task is to instantiate each corresponding Pydantic model based on the action name.

Each model should be instantiated with reasonable placeholder values (or empty defaults) for its required fields.
Use the structure and field names expected by each model. You should return a **list** of model instances,
where each instance is a valid JSON object representing the corresponding Pydantic class.

### Example Input:
instruction = pick up the red cup and move to the table
selected_models = ["PickUpAction", "NavigateAction"]

### Example Output:
[
  {{
    "action_type": "PickUpAction",
    "arm": "left",
    "object": {{
      "type": "cup",
      "color": "red"
    }}
  }},
  {{
    "action_type": "NavigateAction",
    "target_position": {{
      "x": 1.2,
      "y": 0.5,
      "theta": 0.0
    }}
  }}
]

### Notes:
- You **must** use the exact structure and field names of each action model class.
- Use placeholder values where exact data is unknown (e.g., `"object": "type": "box"` or `"target_position": "x": 0.0, "y": 0.0, "theta": 0.0`).
- Wrap all outputs in a single JSON list.
- Do not include any explanation, comments, or extra output. Return **only** the structured list of model instances as JSON.

Now, generate structured action model instances for the following instruction and selected models:
instruction = {instruction}
selected_models = {selected_models}
"""


class ActionNames(BaseModel):
    model_names: List[Literal["PickUpAction", "PlaceAction", "NavigateAction", "SetGripperAction", "LookAtAction",
    "MoveTorsoAction", "GripAction", "ParkArmsAction", "MoveAndPickUpAction", "MoveAndPlaceAction", "OpenAction",
    "CloseAction", "FaceAtAction", "ReachToPickUpAction", "DetectAction", "TransportAction",
    "SearchAction"]] = Field(description="Action model names")

# ActionsTypes = Annotated[
#     Union[
#         MoveTorsoAction, SetGripperAction, GripAction, ParkArmsAction, NavigateAction,
#         PickUpAction, PlaceAction, ReachToPickUpAction, TransportAction, LookAtAction,
#         OpenAction, CloseAction, FaceAtAction, DetectAction, SearchAction,
#         GraspingAction, MoveAndPickUpAction, MoveAndPlaceAction
#     ],
#     Field(discriminator="action_type")
# ]
#
# class Actions(BaseModel):
#     models : List[ActionsTypes] = Field(description="list of instantiated action model instances")

action_classes = [PickUpAction, NavigateAction, PlaceAction, SetGripperAction, LookAtAction,
                  MoveTorsoAction, GripAction, ParkArmsAction, MoveAndPickUpAction, MoveAndPlaceAction,
                  OpenAction, CloseAction, GraspingAction, ReachToPickUpAction, TransportAction,
                    SearchAction, FaceAtAction]

class Actions(BaseModel):
    models : List[Union[*action_classes]] = Field(description="list of instantiated action model instances")

class PyCRAMState(TypedDict):
    action_names : Annotated[list, add_messages]
    action_models : Annotated[list, add_messages]


model_selector_prompt = ChatPromptTemplate.from_template(model_selector_prompt_template)
model_populator_prompt = ChatPromptTemplate.from_template(model_populator_prompt_template)

structured_ollama_llm_pc1 = ollama_llm.with_structured_output(ActionNames, method="json_schema")

structured_ollama_llm_pc2 = ollama_llm.with_structured_output(Actions, method="json_schema")


@tool(description="PyCram Action Designator pydantic model selector tool",
      return_direct=True,)
def model_selector(instruction : str):
    """
    PyCram Action Designator model selector tool that selects relevant Pydantic model names
    based on the input robot task instruction.

    :param instruction: Natural language instruction describing the robot task.
    :return: List of relevant action model class names (as strings).
    """
    print("INSIDE MODEL SELECTOR TOOL")
    print("The instruction is :", instruction)
    answers["instruction"] = instruction
    chain = model_selector_prompt | structured_ollama_llm_pc1
    response = chain.invoke({"input_instruction": instruction})
    # json_response = response.model_dump_json(indent=2, by_alias=True)
    response_python_dict = response.model_dump()
    answers["model_names"] = response_python_dict["model_names"]
    print("response of tool 1 : ", type(response), response)
    # framenet_answers.append(json_response)
    return response_python_dict


class Populater(BaseModel):
    instruction : str = Field()
    model_names : List[str] = Field()

@tool(description="PyCram Action Designator model populator tool that populates Pydantic models",
      return_direct=True, args_schema=Populater)
def model_populator(instruction : str , model_names : List[str]) -> dict :
    """
    PyCram Action Designator model populator tool that populates Pydantic models

    :param instruction: Natural language instruction describing the robot task.
    :param model_names: list of selected pydantic action model class names (as strings) for the robot task.
    :return: dictionary
    """
    print("INSIDE MODEL POPULATOR TOOL")
    instruction_for_populator = {
        "instruction": instruction,
        "selected_models": model_names
    }
    chain = model_populator_prompt | structured_ollama_llm_pc2
    response = chain.invoke(instruction_for_populator)
    # return {"populated_models" : response.models}
    return response.model_dump()

# model_selector_tool_direct_return = Tool.from_function(
#     func=model_selector,
#     name= "model_selector",
#     description= "PyCram Action Designator pydantic model selector tool",
#     return_direct=True
# )

# model_populator_tool_direct_return = Tool.from_function(
#     func=model_populator,
#     name= "model_populator",
#     description= "PyCram Action Designator pydantic model populator tool",
#     return_direct=True  # ✅ This ensures the agent returns it as-is
# )

# model_populator_tool_direct_return = StructuredTool(func=model_populator,
#                name= "model_populator",
#                description= "PyCram Action Designator pydantic model populator",
#                args_schema=Populater,
#                return_direct=True)

# model_populator_tool_direct_return = StructuredTool.from_function(
#     func=model_populator,
#     name= "model_populator",
#     description= "PyCram Action Designator pydantic model populator tool",
#     return_direct=True,
#     args_schema=Populater
# )

# Agent Specific System Prompt
sys_prompt_content = """
    You are a robotic action planning agent that helps convert user instructions into structured robot actions.

    You have access to two tools:

    ---

    ### 🔧 TOOL 1: `model_selector(instruction: str) -> List[str]`

    - Takes a natural language instruction from the user (e.g., "Pick up the red cup and place it on the table")
    - Returns a list of valid action model names, such as:
      ["PickUpActionModel", "PlaceActionModel"]
    - These names correspond to specific Pydantic models for robotic actions
    - Output is strictly limited to known model names

    Use this tool first to decide which action(s) are relevant.

    ---

    ### 🧩 TOOL 2: `model_populator(instruction : str, model_names : List[str] ) -> dict`

    - Takes the user input NL instruction of robot task and a list of selected model names from Tool 1
    - Returns structured and populated Pydantic model instances
    - Each instance will contain all required parameters for execution (e.g., object name, pose, arm, etc.)
    - If something is missing from the original user instruction, use defaults or make reasonable assumptions
    - Do not invent model names not listed in the tool input

    ---

    ### 📝 Goals:

    - Use Tool 1 to **determine** what types of robot actions are needed.
    - Use Tool 2 to **generate structured input** for each action model returned from Tool 1.
    - Your job is to coordinate these tools to map freeform user instructions into a fully specified set of structured robot commands.

    ---

    ### 🧪 Example Workflow:

    #### User:  
    "Move to the table and pick up the bottle"

    #### Tool 1 Output:  
    ["NavigateAction", "PickUpAction"]

    #### Tool 2 Output:  
    [
      {
        "target_location": {
          "position": [1.0, 0.5, 0.0],
          "orientation": [0.0, 0.0, 0.0, 1.0]
        },
        "keep_joint_states": true
      },
      {
        "object_designator": "bottle_1",
        "arm": "left",
        "grasp_description": "front_grasp"
      }
    ]
    
    Only return structured actions at the end of both steps. Follow this tool-based reasoning pipeline strictly.
    If the instruction is unclear, make assumptions but never skip steps.
"""

sys_prompt_content_short = """
    
    You are a robotic action planning agent converting user instructions into structured robot actions using two tools.
    
    ---
    
    ### 🔧 TOOL 1: `model_selector(instruction: str) -> List[str]`
    
    - Input: a natural language task (e.g., "Pick up the red cup and place it on the table")
    - Output: a list of valid model names like ["PickUpActionModel", "PlaceActionModel"]
    - These models represent predefined robotic actions
    - Only return known model names
    
    Always start with this tool to identify required actions.
    
    ---
    
    ### 🧩 TOOL 2: `model_populator(instruction: str, model_names: List[str]) -> dict`
    
    - Input: the same user instruction and the output from Tool 1
    - Output: populated instances of each action model (e.g., object, pose, arm)
    - Use defaults or reasonable assumptions when needed
    - Do not invent new model names
    
    ---
    
    ### 📦 CONTEXT: CRAM-style Action Designators
    
    You may be given **action designators** with detailed info about the object, location, tool, and action.
    
    These are read-only and help you:
    
    - Resolve ambiguities
    - Infer missing parameters
    - Improve reasoning about task context
    
    Never generate or modify them.
    
    ---
    
    ### 📝 Objective:
    
    - Use Tool 1 to select actions.
    - Use Tool 2 to populate structured inputs.
    - Use action designators as **context** only.
    - Produce fully structured robot commands from freeform input.
    
    ---
    
    ### 🧪 Example 1
    
    User: "Move to the table and pick up the bottle"
    
    Tool 1 → ["NavigateAction", "PickUpAction"]
    
    Tool 2 → [
      {
        "target_location": {
          "position": [1.0, 0.5, 0.0],
          "orientation": [0.0, 0.0, 0.0, 1.0]
        },
        "keep_joint_states": true
      },
      {
        "object_designator": "bottle_1",
        "arm": "left",
        "grasp_description": "front_grasp"
      }
    ]
    
    ---
    
    ### 🧪 Example 2
    
    User: "Pick up the bottle near the sink"
    
    Provided Action Designator:
    { "action": {"type": "PickingUp"}, "object": { "name": "Bottle", "properties": {"material": "plastic", "color": "transparent"} }, "location": {"name": "SinkCounter"} }
    
    Tool 1 → ["PickUpAction"]
    
    Tool 2 → { "object_designator": "bottle_1", "arm": "right", "grasp_description": "top_grasp" }
    
    ---
    
    ### 🤖 REMINDERS
    
    - Always follow: Tool 1 → Tool 2
    - Use designators only for reference
    - Never skip tools, even if designators seem complete
    - Fill in missing details with commonsense assumptions
    
    Follow the pipeline strictly.
"""
pycram_agent_sys_prompt = SystemMessage(content=sys_prompt_content)

# Create the agent
pycram_agent = create_agent(ollama_llm, [model_selector, model_populator],
                            agent_sys_prompt=pycram_agent_sys_prompt)

# Agent as Node
def pycram_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    # messages = [
    #                {"role": "system", "content": framenet_system_prompt},
    #            ] + state["messages"]
    result = pycram_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="pycram")
            ]
        },
        goto="supervisor",
    )

# Agent as Node
def pycram_node_pal(state: MessagesState):
    # messages = [
    #                {"role": "system", "content": framenet_system_prompt},
    #            ] + state["messages"]
    result = pycram_agent.invoke(state)
    # print("Pycram agent results: ", type(result),result)
    return {
            "messages": result["messages"][-1]
        }

# Agent as Node
def pycram_node_pal_own(state: MessagesState):
    # messages = [
    #                {"role": "system", "content": framenet_system_prompt},
    #            ] + state["messages"]
    result = pycram_agent.invoke(state)
    # print("Pycram agent results: ", type(result),result)
    return {
            "messages": result["messages"][-1]
        }

if __name__ == '__main__':
    print(Arms.LEFT)
    # print(model_descriptions)

    # responsed = pycram_agent.invoke({"messages" : [HumanMessage(content="generate pycram base models for the instruction pick up the mug from the table")]})
    # print(responsed)

    # chain = model_populator_prompt | structured_ollama_llm_pc2
    #
    # print(chain)
    model_populator.invoke("pick up the cup from the table")