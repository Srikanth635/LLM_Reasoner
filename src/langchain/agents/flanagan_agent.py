from http.client import responses
from typing import Literal
from src.langchain.create_agents import *
from src.langchain.llm_configuration import *
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import Tool
from src.langchain.state_graph import StateModel
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
    # pre_motion_phase: FullPhase
    initiation_phase: InitialPhase
    execution_phase: Phase # Field name 'execution_phase' is different from type 'Phase'
    interaction_phase: Phase # Field name 'interaction_phase' is different from type 'Phase'
    termination_phase: Phase # Field name 'termination_phase' is different from type 'Phase'
    post_motion_phase: Phase # Field name 'post_motion_phase' is different from type 'Phase'

class FinalModel(BaseModel):
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

flanagan_premotion_prompt_template = """
    You are a specialized AI agent, an expert in Robotic Task Initialization. Your sole purpose is to take semi-structured 
    data about a robotic task and transform it into the complete pre_motion_phase of a Flanagan-like task model. 
    You are a data-to-schema transformer.

    *Your Mission:*
    
    You will receive contextual information about a task. Your job is to use this information to generate a single, valid 
    JSON object with one top-level key: pre_motion_phase. This object must contain three sub-sections: goal_definition, predictive_model, and motion_planning.
    
    *Input Inputs You Will Receive:*
    
    - Enriched Attributes JSON: A JSON object containing the action's attributes and their enriched _props dictionaries (e.g., obj_to_be_cut, obj_to_be_cut_props).
     This is your primary source of truth.
    - Action Core: The primary classified action (e.g., "Cutting", "Pouring").
    - Original Instruction: The initial high-level user command, for overall context.
    
    ### Mapping and Generation Instructions ###
    You must meticulously follow these rules to construct the JSON output.
    
    1. Constructing goal_definition:
        This section defines the "what" of the task.
    
        - task (string): Create a concise task summary. Combine the Action Core with the primary object's name (e.g., "Cutting apple", "Pouring water").
        - semantic_annotation (string): Use the Action Core input to create a TaskClass annotation. (e.g., if Action Core is "Cutting", 
        this should be "TaskClass:Manipulation.Cutting").
        - object, target, tool (ObjectModel/ToolModel):
            - Identify the primary object, target, and tool from the keys in the Enriched Attributes JSON (e.g., obj_to_be_cut, goal, utensil).
            - For each of these, create a JSON object with the following structure:
                - id (string): Generate a descriptive, lowercase ID from the object's name (e.g., "apple" becomes "apple_01", "knife" becomes "knife_main").
                - type (string): Use "PhysicalArtifact" for objects/targets and "Gripper" or "Tool" for tools, unless otherwise specified.
                - properties (object): Directly map the corresponding _props dictionary from the input JSON here. For example, the contents of 
                obj_to_be_cut_props become the contents of the properties object.
                - expected_end_state (object): (For object and target only). Define a simple, logical end state based on the task. For a 
                "Cutting" task, the object's end state might be "conditions": "is_cut": true. For a "Pouring" task, the bottle's 
                end state is "conditions": "is_empty": true.
            - Null Handling: If a target or tool is not present in the input data, omit its key entirely from the goal_definition.
            
    2. Constructing predictive_model and motion_planning:
        These sections define the "how." For this task, you should use sensible, generic defaults unless the input data provides specific hints.
        
        - predictive_model:
            - expected_trajectory: Use a generic but descriptive string like "StandardApproachPath" or "LiftingTrajectory".
            - expected_force: Provide reasonable default float values, e.g.,  "initial_N": 1.5, "resistance_range_N": [1.0, 2.0] .
            - confidence_level: Use a high default like 0.95.
            - affordance_model: Infer simple boolean affordances, e.g.,  "tool_affords_cutting": true, "object_affords_being_cut": true .
        - motion_planning:
            - planned_trajectory: Use a default like "GenericPathToTarget".
            - obstacle_avoidance: Use a default like "ReactiveSensorBased".
            - energy_efficiency: Use a default like "Balanced".
        
    ### Output Specification ###
    - Your response MUST be a single, valid JSON object.
    - The only top-level key must be pre_motion_phase.
    - Do not include any other keys, text, explanations, or markdown formatting.
    
    Example
    Input Action Core: Cutting
    
    Input Enriched Attributes JSON:
    
    JSON
    
    
      "obj_to_be_cut": "apple",
      "obj_to_be_cut_props": 
        "color": "red",
        "shape": "round",
        "texture": "smooth"
      ,
      "utensil": "knife",
      "utensil_props": 
        "material": "steel",
        "edge": "sharp"
      
    
    Your Output:
    
    JSON
    
    
      "pre_motion_phase": 
        "goal_definition": 
          "task": "Cutting apple",
          "semantic_annotation": "TaskClass:Manipulation.Cutting",
          "object": 
            "id": "apple_01",
            "type": "PhysicalArtifact",
            "properties": 
              "color": "red",
              "shape": "round",
              "texture": "smooth"
            ,
            "expected_end_state": 
              "conditions": 
                "is_cut": true
              
            
          ,
          "tool": 
            "id": "knife_main",
            "type": "Tool",
            "properties": 
              "material": "steel",
              "edge": "sharp"
            
          
        ,
        "predictive_model": 
          "expected_trajectory": "StandardCuttingApproach",
          "expected_force": 
            "initial_N": 2.0,
            "resistance_range_N": [
              1.5,
              3.0
            ]
          ,
          "confidence_level": 0.95,
          "affordance_model": 
            "tool_affords_cutting": true,
            "object_affords_being_cut": true
          
        ,
        "motion_planning": 
          "planned_trajectory": "DirectPathToObject",
          "obstacle_avoidance": "ReactiveSensorBased",
          "energy_efficiency": "Balanced"
        
      
    ---
    
    Now perform the task for given action, enriched_attributes and instruction:
    
    Action Core: {action_core}
    Enriched Attributes: {enriched_attributes}
    Original Instruction: {instruction}
    
"""
flanagan_premotion_prompt = ChatPromptTemplate.from_template(flanagan_premotion_prompt_template)

flanagan_phaser_prompt_template = """
    You are an expert AI in Robotic Motion Phase Decomposition. Your sole purpose is to generate the sequential motion phases of a
    robotic task (Initiation through PostMotion) by decomposing a task's goal into a logical series of steps, based on a wealth of provided context.

    *Your Mission:*
    
    You will receive a comprehensive set of inputs that define a task's goal and low-level structure. Your job is to use this information
    to reason about the physical steps required to perform the action and to output a valid JSON object describing the five sequential motion phases.
    
    ### Inputs You Will Receive: ###
    
    - Pre-Motion Phase JSON: The detailed definition of the task's goal, including the specific ids and properties of the object,
        target, and tool. This is your primary source of truth for "what" is being manipulated.
    - CRAM Plan: A low-level symbolic plan that provides a strong hint about the fundamental sequence of actions.
    - Action Core: The primary classified action (e.g., "Cutting", "Placing").
    - Original Instruction: The initial high-level user command, for overall context.
    
    Core Logic and Generation Instructions
    To construct the output, you must synthesize information from all inputs and follow this logic:
    
    1. Use the CRAM Plan as a Scaffold:
        The CRAM Plan provides an excellent blueprint for your SubPhases. Analyze its action types (e.g., grab-object, lift-object, 
        put-object) and use them to define the names and sequence of your SubPhases within the appropriate main phases.
    
    2. Reference the Pre-Motion Data:
        Constantly refer to the Pre-Motion Phase JSON to get the correct ids for the object, tool, and target. These IDs are essential
        for creating meaningful goalState conditions. For example, a goal state should look like: "goalState":  "ObjectState":  "apple_01.IsHeld": true  .
    
    3. Logical Phase Breakdown:
        Structure your decomposition into the five phases. A typical task flow is as follows:
    
        - initiation_phase: The actions required to begin the task, such as moving the robot's arm to the object (Reach), 
        preparing the gripper (PreGrasp), and securing the object (Grasp).
        - execution_phase: The actions of moving the secured object to the point of interaction. This could involve Lift, Transport, and Align sub-phases.
        - interaction_phase: The main event where the tool interacts with the object or target. This is the core of the task, such as the Cut, Pour, or Press sub-phase.
        - termination_phase: The actions to conclude the interaction, such as retracting the tool (Retract), placing the object (Place), and releasing it (Release).
        - post_motion_phase: Final clean-up actions, like returning the arm to a neutral "home" position (ReturnToHome) and 
        updating the internal world model (UpdateKnowledgeBase).

    4. Define goalState and SymbolicGoals:

        - The goalState for each SubPhase must define the specific, verifiable condition that marks its completion.
        - The SymbolicGoals for each main phase should list the high-level outcomes achieved during that phase.
    
    ### Output Specification ###
    - Your response MUST be a single, valid JSON object.
    - The JSON object should contain only the five sequential phase keys: initiation_phase, execution_phase, interaction_phase, termination_phase, and post_motion_phase.
    - Do not include any other keys, text, explanations, or markdown formatting outside of the JSON structure.
    
    ### Example: ###
    Input Action Core: Cutting
    
    Input CRAM Plan: (perform (an action (type cut-object) (an object (type apple)) (utensil (an object (type knife)))))
    
    Input Pre-Motion Phase JSON:
    
    JSON
    
    
      "pre_motion_phase": 
        "goal_definition": 
          "task": "Cutting apple",
          "object":  "id": "apple_01", "properties":  "color": "red" ,
          "tool":  "id": "knife_main", "properties":  "edge": "sharp" 
        
      
    
    Your Output:
    
    JSON
    
    
      "initiation_phase": 
        "name": "Initiation",
        "SubPhases": [
          
            "name": "ReachForTool",
            "description": "Move end-effector towards the knife.",
            "goalState":  "ToolState":  "knife_main.IsInRange": true 
          ,
          
            "name": "GraspTool",
            "description": "Grasp the knife using the gripper.",
            "goalState":  "ToolState":  "knife_main.IsHeld": true 
          
        ],
        "SymbolicGoals": [  "goal": "ToolAcquired"  ],
        "SemanticAnnotation": "PhaseClass:Initiation"
      ,
      "execution_phase": 
        "name": "Execution",
        "SubPhases": [
          
            "name": "ApproachObjectWithTool",
            "description": "Move the knife into position above the apple.",
            "goalState":  "ObjectState":  "apple_01.IsInRange": true 
          ,
          
            "name": "AlignToolForCut",
            "description": "Align the knife edge with the apple surface.",
            "goalState":  "ToolState":  "knife_main.IsAligned": true 
          
        ],
        "SymbolicGoals": [  "goal": "ReadyToInteract"  ],
        "SemanticAnnotation": "PhaseClass:Execution"
      ,
      "interaction_phase": 
        "name": "Interaction",
        "SubPhases": [
          
            "name": "PerformCut",
            "description": "Apply downward force with the knife to slice the apple.",
            "goalState":  "ObjectState":  "apple_01.IsCut": true  
          
        ],
        "SymbolicGoals": [  "goal": "ObjectStateModified"  ],
        "SemanticAnnotation": "PhaseClass:Interaction"
      ,
      "termination_phase": 
        "name": "Termination",
        "SubPhases": [
          
            "name": "RetractTool",
            "description": "Lift the knife away from the cut apple.",
            "goalState":  "ToolState":  "knife_main.IsClear": true  
          ,
          
            "name": "ReleaseTool",
            "description": "Place the knife back in a safe location and release grip.",
            "goalState":  "ToolState": "knife_main.IsHeld": false  
          
        ],
        "SymbolicGoals": [  "goal": "InteractionComplete"  ],
        "SemanticAnnotation": "PhaseClass:Termination"
      ,
      "post_motion_phase": 
        "name": "PostMotion",
        "SubPhases": [
          
            "name": "ReturnToHome",
            "description": "Move the end-effector to a neutral home position.",
            "goalState":  "RobotState":  "arm.IsAtHome": true  
          ,
          
            "name": "UpdateWorldModel",
            "description": "Update the knowledge base with the new state of the apple.",
            "goalState":  "SystemState":  "KnowledgeBase.IsUpdated": true  
          
        ],
        "SymbolicGoals": [  "goal": "TaskComplete"  ],
        "SemanticAnnotation": "PhaseClass:PostMotion"
      
    ---
    
    Now, perform the task for the given premotion_phase info, CRAM plan, action core and instruction:
    
    Pre-Motion Phase JSON: {premotion_phase}
    CRAM Plan: {cram_plan}
    Action Core: {action_core}
    Original Instruction: {instruction}
    
"""
flanagan_phaser_prompt = ChatPromptTemplate.from_template(flanagan_phaser_prompt_template)

flanagan_combiner_prompt_template = """
    You are a simple, hyper-focused JSON Manipulation Utility. Your only function is to combine two separate JSON objects into a single,
    unified JSON object. You do not analyze, interpret, or change the content in any way.
    
    
    *Your Mission:*
    
    You will be given two JSON objects, JSON_Object_A and JSON_Object_B. You must merge them according to the rules below.
    
    ### Execution Rules: ###
    
    - Combine Keys: Take all the top-level keys from JSON_Object_A and all the top-level keys from JSON_Object_B and place them together in a new, single JSON object.
    - Add task Key: Add a new top-level key named task. Its value should be a concise summary of the task, which you can find in JSON_Object_A at the path pre_motion_phase.goal_definition.task.
    - Do Not Modify: You must not alter, add, or remove any content or keys within the original objects. Your only job is to merge them at the top level.
    - Ensure Validity: The final output must be a single, syntactically perfect JSON object.
    
    ### Output Specification ###
    - Your response MUST be only the final, merged JSON object.
    - Do not include any text, explanations, or markdown formatting like ```json.
    
    ### Example ###
    Input JSON_Object_A:
    
    JSON
    
    
      "pre_motion_phase": 
        "goal_definition": 
          "task": "Cutting apple",
          "object":  "id": "apple_01" 
        
      
    
    Input JSON_Object_B:
    
    JSON
    
    
      "initiation_phase": 
        "name": "Initiation",
        "SubPhases": [  "name": "GraspTool"  ]
      ,
      "execution_phase": 
        "name": "Execution"
      
    
    Your Output:
    
    JSON
    
    
      "task": "Cutting apple",
      "pre_motion_phase": 
        "goal_definition": 
          "task": "Cutting apple",
          "object":  "id": "apple_01" 
        
      ,
      "initiation_phase": 
        "name": "Initiation",
        "SubPhases": [  "name": "GraspTool"  ]
      ,
      "execution_phase": 
        "name": "Execution"
      
    ---
    
    Now, perform the task of merging on given data,
    
    JSON_Object_A: {premotion_phase}
    JSON_Object_B: {phaser}
"""

flanagan_combiner_prompt = ChatPromptTemplate.from_template(flanagan_combiner_prompt_template)

# llm_fn = ChatOpenAI(model="gpt-4o-mini")
# structured_llm_fn = llm.with_structured_output(TaskModel)
structured_ollama_llm_fl = ollama_llm.with_structured_output(TaskModel, method="json_schema")



def flanagan_premotion_node(state : StateModel):
    """
    Generate a structured Flanagan-style task premotion phase model representation from a natural language instruction, action core
    and enriched attributes information
    """
    print("INSIDE flanagan PREMOTION TOOL")
    instruction = state['instruction']
    action_core = state['action_core']
    enriched_json_attributes = state['enriched_action_core_attributes']

    chain = flanagan_premotion_prompt | ollama_llm_small.with_structured_output(FullPhase, method="json_schema")
    response = chain.invoke({"instruction": instruction, "action_core" : action_core, "enriched_attributes" : enriched_json_attributes})
    print(response)
    return {'premotion_phase' : response.model_dump_json(indent=2, by_alias=True)}

def flanagan_phaser_node(state:StateModel):
    """
    Generate a structured Flanagan-style task phase model representation from a natural language instruction, action core,
    enriched attributes information and low level cram action designator
    """
    print("INSIDE flanagan PHASER TOOL")
    premotion_phase = state['premotion_phase']
    cram_plan = state['cram_plan_response']
    action_core = state['action_core']
    instruction = state['instruction']

    chain2 = flanagan_phaser_prompt | ollama_llm_small.with_structured_output(TaskModel, method="json_schema")
    response2 = chain2.invoke({"premotion_phase": premotion_phase, "cram_plan" : cram_plan, "action_core" : action_core, "instruction" : instruction})
    print(response2)
    return {'phaser' : response2.model_dump_json(indent=2)}

def flanagan_repr(state : StateModel):
    """
    Generate a structured full Flanagan-style task model representation from a predefined premotion phase and remaining
    sequential phase information.
    """
    print("INSIDE flanagan REPR")
    premotion_phase = state['premotion_phase']
    phaser = state['phaser']

    chain3 = flanagan_combiner_prompt | ollama_llm_small.with_structured_output(FinalModel, method="json_schema")
    response3 = chain3.invoke({"premotion_phase": premotion_phase, "phaser" : phaser})
    print(response3)
    return {'flanagan' : response3.model_dump_json(indent=2)}


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
    return_direct=True
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

def flanagan_node_pal(state: MessagesState):
    result = flanagan_agent.invoke(state)
    return {
            "messages": result["messages"][-1]
        }


if __name__ == "__main__":
    print("INSIDE MAIN")

    # test_flanagan()
    # print(flanagan_tool("pick up the bottle from the sink"))
    # print(fnr[0])
    # flanagan_agent.invoke({'messages' : [HumanMessage(content='pick up the bottle from the sink')]})

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

    instruction = "pick up the cooking pan from the wooden drawer"
    action_core = "PickingUp"
    enriched_json_attributes = """
         {  "obj_to_be_grabbed": "cooking pan",  "obj_to_be_grabbed_props": {    "material": "metal",    "shape": "rectangular",    "handle": "present"  },  "action_verb": "pick up",  "location": "wooden drawer",  "location_props": {    "material": "wood",   "shape": "rectangular" }}
    """
    cram_plan_ex =  '(perform (an action (type grab-object) (an object (type cooking pan) (material metal) (shape rectangular) (size medium)) (source (a location (on wooden drawer) (material wood) (shape rectangular) (type drawer)))))'

    premotion_phase_ex = """
        goal_definition=GoalDefinition(task='PickingUp cooking pan', semantic_annotation='TaskClass:Manipulation.PickingUp', 
        object=ObjectModel(id='cooking_pan_01', type='PhysicalArtifact', properties=ObjectProperties(size=None, texture=None, 
        material='metal', fill_level=None, contents=None, hardness=None, friction_coefficient=None, elasticity=None, strain_limit=1000000000000000.0), 
        expected_end_state=GoalState(conditions={'is_grabbed': {'is_grabbed': True}})), target=None, tool=ToolModel(id='gripper_main', 
        type='Gripper', properties={})) predictive_model=PredictiveModel(expected_trajectory='StandardApproachPath', 
        expected_force={'initial_N': 1.5, 'resistance_range_N': [1.0, 2.0]}, confidence_level=0.95, affordance_model=
        {'tool_affords_grabbing': True, 'object_affords_being_grabbed': True}) motion_planning=MotionPlanning(planned_trajectory='GenericPathToTarget', 
        obstacle_avoidance='ReactiveSensorBased', energy_efficiency='Balanced')
    """

    phaser_ex = """
        task='PickingUp cooking pan' initiation_phase=InitialPhase(initial_state=InitialState(robot_pose={'arm': [0.4, 0.2, 0.5], 
        'gripper': [0.5, 0.5, 0.5]}, tool_position=[0.3, 0.2, 0.4], target_object_position=[0.6, 0.3, 0.4]), motion_initialization=MotionInitialization
        (joint_activation={'arm': 0.4, 'gripper': 0.3}, velocity_profile='slow', motion_priming={'gripper_state': False, 'gripper_position': False}), 
        SubPhases=[SubPhase(name='ReachForObject', description='Move end-effector towards the cooking pan.', goalState=[GoalState(conditions={'ObjectState': 
        {'cooking_pan_01.IsInRange': True}})]), SubPhase(name='PreGrasp', description='Prepare the gripper to grasp the cooking pan.', 
        goalState=[GoalState(conditions={'GripperState': {'gripper_main.IsOpen': False}})]), SubPhase(name='GraspObject', 
        description='Secure the grip on the cooking pan.', goalState=[GoalState(conditions={'ObjectState': {'cooking_pan_01.IsGrabbed': True}})])], 
        SymbolicGoals=[GoalState(conditions={'goal': {'is_grabbed': True}})], SemanticAnnotation='PhaseClass:Initiation') 
        execution_phase=Phase(SubPhases=[SubPhase(name='LiftObject', description='Lift the cooking pan from the drawer.', 
        goalState=[GoalState(conditions={'ObjectState': {'cooking_pan_01.IsLifted': True}})]), SubPhase(name='Transport', 
        description='Move the cooking pan to a stable position.', goalState=[GoalState(conditions={'ObjectState': {'cooking_pan_01.IsStable': True}})])], 
        SymbolicGoals=[GoalState(conditions={'goal': {'is_lifted': True}})], SemanticAnnotation='PhaseClass:Execution') 
        interaction_phase=Phase(SubPhases=[SubPhase(name='ConfirmGrasp', description='Confirm that the cooking pan is securely grasped.', 
        goalState=[GoalState(conditions={'ObjectState': {'cooking_pan_01.IsSecure': True}})])], 
        SymbolicGoals=[GoalState(conditions={'goal': {'is_secure': True}})], SemanticAnnotation='PhaseClass:Interaction') 
        termination_phase=Phase(SubPhases=[SubPhase(name='ReleaseObject', description='Release the cooking pan from the gripper.', 
        goalState=[GoalState(conditions={'ObjectState': {'cooking_pan_01.IsReleased': True}})]), SubPhase(name='RetractGripper', 
        description='Move the gripper to a safe position.', goalState=[GoalState(conditions={'GripperState': {'gripper_main.IsOpen': True}})])], 
        SymbolicGoals=[GoalState(conditions={'goal': {'is_released': True}})], SemanticAnnotation='PhaseClass:Termination') 
        post_motion_phase=Phase(SubPhases=[SubPhase(name='ReturnToHome', description='Return the end-effector to the home position.', 
        goalState=[GoalState(conditions={'RobotState': {'arm.IsAtHome': True}})]), SubPhase(name='UpdateWorldModel', 
        description='Update the world model to reflect the new state of the cooking pan.', goalState=[GoalState(conditions={'SystemState': 
        {'KnowledgeBase.IsUpdated': True}})])], SymbolicGoals=[GoalState(conditions={'goal': {'task_complete': True}})], SemanticAnnotation='PhaseClass:PostMotion')
    """
    # flanagan_premotion_tool(instruction=instruction, action_core=action_core, enriched_json_attributes=enriched_json_attributes)
    # flanagan_phaser_tool(premotion_phase=premotion_phase_ex, cram_plan=cram_plan_ex, action_core=action_core, instruction=instruction)
    flanagan_repr(premotion_phase=premotion_phase_ex, phaser=phaser_ex)