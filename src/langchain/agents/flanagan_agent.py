from http.client import responses
from typing import Literal
from src.langchain.create_agents import *
from src.langchain.llm_configuration import *
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.agents import Tool

from langgraph.graph import MessagesState
from langgraph.types import Command

import os


###############################Structured Output flanagan START#######################################################

from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field

# ... (Keep your existing GoalState, SubPhase, ObjectProperties, ObjectModel, ToolModel)
# ... (PredictiveModel, MotionPlanning, InitialState, MotionInitialization also remain unchanged at their definition)

class GoalState(BaseModel):
    conditions: Dict[str, Dict[str, bool]] = Field(
        description="Dictionary of symbolic conditions (e.g., {'toolstate': {'grasped': true}} or {'objectstate': {'held': true}}"
                    "or {'goal_state' : { 'emptied_into_target': true }})"
    )


class SubPhase(BaseModel):
    name: str = Field(description="Name of subphase (e.g., Reach, Grasp, Cut)")
    description: str = Field(description="Short explanation of what the subphase does")
    goalState: List[GoalState] = Field(description="Symbolic goal state for this subphase")


class ObjectProperties(BaseModel):
    size: Optional[str] = Field(default=None, description="Size (e.g., small, medium)")
    texture: Optional[str] = Field(default=None, description="Surface texture (e.g., smooth, rough)")
    material: Optional[str] = Field(default=None, description="Material type (e.g., plastic, wood)")
    fill_level: Optional[str] = Field(default=None, description="Fill status if applicable (e.g., full, half)")
    contents: Optional[str] = Field(default=None, description="Contents inside the object (e.g., water)")
    hardness: Optional[float] = Field(default=None, description="Used for cutting tasks")
    friction_coefficient: Optional[float] = Field(default=None, description="Surface resistance")
    elasticity: Optional[float] = Field(default=None, description="Material deformation measure")
    strain_limit: Optional[float] = Field(default=None, description="Limit before breaking or slicing")


class ObjectModel(BaseModel):
    id: str = Field(description="Object identifier")
    type: str = Field(description="Type/category of the object")
    properties: ObjectProperties = Field(description="All semantic and physical properties")
    expected_end_state: Optional[GoalState] = Field(default=None, description="Goal state after task")


class ToolModel(BaseModel):
    id: str = Field(description="Tool ID")
    type: str = Field(description="Tool type (e.g., gripper, knife)")
    properties: Dict[str, Union[str, float]] = Field(description="Attributes such as grip force or sharpness")


class GoalDefinition(BaseModel):
    task: str = Field(description="Top-level task label")
    semantic_annotation: str = Field(description="Ontology-aligned task class")
    object: ObjectModel = Field(description="Primary object to manipulate")
    target: Optional[ObjectModel] = Field(default=None, description="Target object or destination, if applicable")
    tool: ToolModel = Field(description="Tool used in the task")


class PredictiveModel(BaseModel):
    expected_trajectory: str = Field(description="Motion path name or label")
    expected_force: Dict[str, Union[float, List[float]]] = Field(description="Force requirements (init, range)")
    confidence_level: float = Field(description="Model confidence from 0.0 to 1.0")
    affordance_model: Dict[str, bool] = Field(description="Boolean labels for supported actions")


class MotionPlanning(BaseModel):
    planned_trajectory: str = Field(description="Path the robot plans to follow")
    obstacle_avoidance: str = Field(description="Type of collision or obstacle strategy")
    energy_efficiency: str = Field(description="Power-saving or effort optimization approach")


class InitialState(BaseModel):
    robot_pose: Dict[str, List[float]] = Field(description="Robot's base pose and orientation")
    tool_position: List[float] = Field(description="Tool starting position")
    target_object_position: List[float] = Field(description="Target object starting position")


class MotionInitialization(BaseModel):
    joint_activation: Dict[str, float] = Field(description="Joint angles or activation levels")
    velocity_profile: str = Field(description="Velocity pattern (e.g., ramp-up)")
    motion_priming: Dict[str, bool] = Field(description="System checks or readiness flags")


# Corrected FullPhase
class FullPhase(BaseModel):
    goal_definition: GoalDefinition = Field(description="Top-level task and planning context")
    predictive_model: PredictiveModel = Field(description="Models of expected control behavior")
    motion_planning: MotionPlanning = Field(description="Execution strategies and trajectory")


# Corrected InitialPhase
class InitialPhase(BaseModel):
    initial_state: InitialState = Field(description="Initial spatial configuration")
    motion_initialization: MotionInitialization = Field(description="Motor and joint configuration")
    SubPhases: List[SubPhase] = Field(description="Action primitives during initiation") # This was okay
    SymbolicGoals: List[GoalState] = Field(description="Completion criteria for the phase") # This was okay
    SemanticAnnotation: str = Field(description="Ontology label like PhaseClass:Initiation") # This was okay


# Phase model (was okay, but good to be consistent if it's a main phase component)
# If 'Phase' here is meant to be a generic phase structure used multiple times, its field names are fine.
# However, if 'Phase' itself is used as a type for a field name 'Phase' in TaskModel, that specific field in TaskModel needs changing.
class Phase(BaseModel):
    SubPhases: List[SubPhase] = Field(description="Symbolic breakdown of this motion phase")
    SymbolicGoals: List[GoalState] = Field(description="Expected symbolic goal of the phase")
    SemanticAnnotation: Optional[str] = Field(default=None, description="Ontology-compatible phase class")


# Corrected TaskModel
class TaskModel(BaseModel):
    task: str = Field(description="Label for the task (e.g., PourWaterFromBottle)")
    pre_motion_phase: FullPhase
    initiation_phase: InitialPhase
    execution_phase: Phase # Field name 'execution_phase' is different from type 'Phase'
    interaction_phase: Phase # Field name 'interaction_phase' is different from type 'Phase'
    termination_phase: Phase # Field name 'termination_phase' is different from type 'Phase'
    post_motion_phase: Phase # Field name 'post_motion_phase' is different from type 'Phase'




###############################Structured Output flanagan END#########################################################

flanagan_answers = []

flanagan_prompt_template = """
    
    You are an expert AI in robotics task planning and semantic task representation. For each instruction given, generate a Flanagan-like JSON representation. 
    This JSON object must strictly conform to a predefined Pydantic schema representing a complete robotic task (TaskModel).

    The overall structure is a `TaskModel` which includes several phases: `PreMotionPhase`, `InitiationPhase`, `ExecutionPhase`, `InteractionPhase`, `TerminationPhase`, and `PostMotionPhase`. 
    Each phase can contain `SubPhases` and `SymbolicGoals`. The `PreMotionPhase` is crucial as it defines the overall `GoalDefinition` including the primary `object`, `target` (if any), and `tool`.
    
    **Key Instructions for JSON Generation:**
    
    **Output Format**
        Generate a single valid JSON object.
        Do not use YAML or any other format.
        Do not include any explanatory text outside the JSON structure.
    **Schema Adherence**
        ObjectModel must contain:
            id
            type
            properties (e.g., size, material, hardness)
            expected_end_state
        ToolModel must contain:
            id
            type
            properties (e.g., grip_force, sharpness)
        GoalState defines symbolic conditions such as:
            ObjectState: <object_id>.IsHeld = True
            ToolState: <tool_id>.Holding = <object_id>
        SubPhase includes:
            name
            description
            goalState
    **ID Generation**
        Use descriptive lowercase IDs for id fields in ObjectModel and ToolModel:
            Example: "bottle1", "knife_main", "table_surface"
    **Inference and Defaults**
        If instruction lacks specific details:
            Infer reasonable values or use placeholders like:
                "to_be_defined"
                "generic_material"
                null for optional fields
        For physical properties:
            Estimate float values if known (e.g., apple = soft ~0.3, metal block = hard ~0.9)
            Use null if unknown
        For PredictiveModel, MotionPlanning, InitialState, MotionInitialization:
            Use typical robotic defaults:
                expected_trajectory: "StandardApproachPath"
                obstacle_avoidance: "ReactiveSensorBased"
                robot_pose: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    **Phase and SubPhase Breakdown**
        Infer logical sequence of main phases based on instruction.
        Break down each phase into SubPhase objects:
            InitiationPhase
            ExecutionPhase
            InteractionPhase
            TerminationPhase
        Each SubPhase should have:
            name
            description
            goalState
        Example for "picking up":
            ReachToPreGrasp
            ApproachObject
            Grasp
            Lift
    **Semantic Annotations**
        Use ontology-style labels:
            GoalDefinition.semantic_annotation: "TaskClass:Manipulation.Take"
            Phase.SemanticAnnotation: "PhaseClass:Motion.Initiation"
    **Empty/Optional Fields**
        If a field is optional and not applicable:
            Use null (for JSON)
        For optional complex objects (like target in GoalDefinition):
            Omit them or set to null if not present
    **Task Label**
        Top-level TaskModel.task should be PascalCase summary:
            Example: "PickUpBottleFromSink"
    
    ---
    
    Example 1:
    
    #Input
    Instruction : Pour water from the bottle into the container
    
    #Output
    
    "task": "Pour water from the bottle into the container",
    "pre_motion_phase": 
      "goal_definition": 
        "task": "Pour Water",
        "semantic_annotation": "TaskClass:Pouring",
        "object": 
          "id": "bottle_01",
          "type": "PhysicalArtifact",
          "properties": 
            "contents": "water",
            "fill_level": "full",
            "material": "plastic"
          "expected_end_state": 
            "conditions": "goal_state" : "conditions" :  "emptied_into_target": true 
        "target": 
          "id": "container_01",
          "type": "PhysicalArtifact",
          "properties": 
            "capacity_ml": 500,
            "initial_contents": "empty"
        "tool": 
          "id": "gripper_standard",
          "type": "Gripper",
          "properties": 
            "grip_type": "cylindrical",
            "max_force": 8
      "predictive_model": 
        "expected_trajectory": "Trajectory:TiltArcToContainer",
        "expected_force": 
          "initial_N": 1.0,
          "resistance_range_N": [0.8, 1.5]
        "confidence_level": 0.96,
        "affordance_model": 
          "tool_affords_holding": true,
          "object_affords_pouring": true
      "motion_planning": 
        "planned_trajectory": "Trajectory:TiltPourArc",
        "obstacle_avoidance": "Basic",
        "energy_efficiency": "Optimized"
    "initiation_phase": 
      "initial_state": 
        "robot_pose": 
          "position": [0.0, 0.0, 0.0],
          "orientation": [0.0, 0.0, 0.0, 1.0]
        "tool_position": [0.3, 0.0, 0.2],
        "target_object_position": [0.4, 0.1, 0.0]
      "motion_initialization": 
        "joint_activation": 
          "joint1": 25,
          "joint2": 35
        "velocity_profile": "Profile:LinearRampUp",
        "motion_priming": 
          "pregrasp_pose_reached": true,
          "tool_ready": true
      "SubPhases": [
    
          "name": "Reaching",
          "description": "Move end-effector toward the bottle",
          "goalState": ["conditions" :  "arm_state":  "aligned": true  ]
    
          "name": "Grasping",
          "description": "Grasp the bottle using gripper",
          "goalState": ["conditions" :  "tool_state":  "grasped": true  ]
      ],
      "SymbolicGoals": [
        "conditions" :  "arm_state":  "aligned": true  ,
        "conditions" :  "tool_state":  "grasped": true  ,
        "conditions" :  "gripper_status":  "engaged": true  
      ],
      "SemanticAnnotation": "PhaseClass:Initiation"
    "execution_phase": 
      "SubPhases": [
        
          "name": "AlignToolWithTarget",
          "description": "Align bottle above the container",
          "goalState": [
            "conditions" :  "tool_state":  "aligned": true  
          ]
    
          "name": "Approaching",
          "description": "Move bottle into pouring position",
          "goalState": [
            "conditions" : "tool_state":  "engaged": true ,
            "conditions" : "target_object_state":  "contacted": true 
          ]
        
      ],
      "SymbolicGoals": [
        "conditions" : "tool_state":  "ready_to_pour": true ,
        "conditions" : "target_object_state":  "ready_to_receive": true 
      ],
      "SemanticAnnotation": "PhaseClass:Execution"
    "interaction_phase": 
      "SubPhases": [
        
          "name": "Pouring",
          "description": "Tilt bottle to pour water",
          "goalState": ["conditions" :  "flow_state":  "initiated": true  ]
        
          "name": "MonitorAndControl (or) MonitoringJointState",
          "description": "Adjust tilt and monitor flow",
          "goalState": ["conditions" :  "control_state":  "stable": true  ]
    
          "name": "StopPour (or) Orienting",
          "description": "Return bottle to upright",
          "goalState": ["conditions" :  "flow_state":  "stopped": true  ]
        
      ],
      "SymbolicGoals": [
        "conditions" : "task_status":  "pouring": true 
      ],
      "SemanticAnnotation": "PhaseClass:Interaction"
    "termination_phase": 
      "SubPhases": [
        
          "name": "Orienting",
          "description": "Return bottle to upright position",
          "goalState": ["conditions" :  "orientation_status":  "upright": true  ]
    
          "name": "Placing (and) Releasing",
          "description": "Place bottle back and release grip",
          "goalState": ["conditions" :  "gripper_status":  "released": true  ]
        
      ],
      "SymbolicGoals": [
        "conditions" : "object_state":  "placed": true ,
        "conditions" : "gripper_status":  "released": true 
      ],
      "SemanticAnnotation": "PhaseClass:Termination"
    "post_motion_phase": 
      "SubPhases": [
        
          "name": "UpdateStatus",
          "description": "Record task completion and update world model",
          "goalState": ["conditions" :  "knowledge_base":  "updated": true  ]
    
          "name": "PrepareNextTask",
          "description": "Reset for next operation",
          "goalState": ["conditions" :  "system_state":  "initialized": true  ]
        
      ],
      "SymbolicGoals": [
        "conditions" : "system_state":  "ready": true 
      ],
      "SemanticAnnotation": "PhaseClass:PostMotion"
    
    
    ---
    
    Now, given the instruction below, produce a similar output:
    
    Instruction: {input_instruction}
"""

flanagan_prompt = ChatPromptTemplate.from_template(flanagan_prompt_template)

# llm_fn = ChatOpenAI(model="gpt-4o-mini")
structured_llm_fn = llm.with_structured_output(TaskModel)
structured_ollama_llm_fl = ollama_llm.with_structured_output(TaskModel, method="json_schema")


# Agent Specific Tools
@tool
def flanagan_tool(instruction: str):
    """
    Generate a structured Flanagan-style task model representation from a natural language instruction.

    This tool uses a language model (LLM) with structured output based on the `TaskModel` schema to produce
    a comprehensive, ontology-compatible JSON representation of robot task phases. It segments the action
    into phases like PreMotion, Initiation, Execution, Interaction, Termination, and PostMotion.

    The LLM output adheres to a validated Pydantic model, enabling downstream reasoning, execution,
    or integration with robotic planning systems.

    Args:
        instruction (str): A natural language instruction (e.g., "Pour water from the bottle into the container")

    Returns:
        str: Flanagan representation of the input instruction.
    """
    print("INSIDE flanagan TOOL")
    chain = flanagan_prompt | structured_ollama_llm_fl
    response = chain.invoke({"input_instruction": instruction})
    json_response = response.model_dump_json(indent=2, by_alias=True)
    flanagan_answers.append(json_response)
    return json_response


flanagan_tool_direct_return = Tool.from_function(
    func=flanagan_tool,
    name= "flanagan_tool",
    description= "flanagan representation of the input string with direct return tool output",
    return_direct=True  # ✅ This ensures the agent returns it as-is
)


# Agent
flanagan_agent = create_agent(ollama_llm, [flanagan_tool_direct_return])


# Agent as Node
def flanagan_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = flanagan_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="flanagan")
            ]
        },
        goto="supervisor",
    )


if __name__ == "__main__":
    print("INSIDE MAIN")

    # test_flanagan()
    # print(flanagan_tool("pick up the bottle from the sink"))
    # print(fnr[0])
    flanagan_agent.invoke({'messages' : [HumanMessage(content='pick up the bottle from the sink')]})

    # @tool
    # def flans_tool(instruction: str):
    #     """
    #     Generate only Frame name of flanagan for a given natural language instruction.
    #
    #     This function invokes a prompt chain using a predefined flanagan prompt template and a language model
    #     function (llm_fn) to extract frame semantic frame name from the input instruction.
    #
    #     Args:
    #         a (str): A natural language instruction (e.g., "Pick up the bottle from the sink").
    #
    #     Returns:
    #         str : frame name suitable for given input instruction.
    #     """
    #     print("INSIDE FRAME TOOL")
    #     chain = flanagan_prompt | structured_llm_fn
    #     response = chain.invoke({"input_instruction": instruction})
    #     json_response = response.model_dump_json(indent=2, by_alias=True)
    #     # return "Srikanth"
    #     return response