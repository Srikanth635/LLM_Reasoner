from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field


class GoalState(BaseModel):
    __root__: Dict[str, Dict[str, bool]]  # e.g., {"ObjectState": {"Grasped": True}}


class SubPhase(BaseModel):
    name: str = Field(description="Name of subphase (e.g., Reach, Grasp, Cut)")
    description: str = Field(description="Short explanation of what the subphase does")
    goalState: GoalState = Field(description="Symbolic goal state for this subphase")


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


class FullPhase(BaseModel):
    GoalDefinition: GoalDefinition = Field(description="Top-level task and planning context")
    PredictiveModel: PredictiveModel = Field(description="Models of expected control behavior")
    MotionPlanning: MotionPlanning = Field(description="Execution strategies and trajectory")


class InitialPhase(BaseModel):
    InitialState: InitialState = Field(description="Initial spatial configuration")
    MotionInitialization: MotionInitialization = Field(description="Motor and joint configuration")
    SubPhases: List[SubPhase] = Field(description="Action primitives during initiation")
    SymbolicGoals: GoalState = Field(description="Completion criteria for the phase")
    SemanticAnnotation: str = Field(description="Ontology label like PhaseClass:Initiation")


class Phase(BaseModel):
    SubPhases: List[SubPhase] = Field(description="Symbolic breakdown of this motion phase")
    SymbolicGoals: GoalState = Field(description="Expected symbolic goal of the phase")
    SemanticAnnotation: Optional[str] = Field(default=None, description="Ontology-compatible phase class")


class TaskModel(BaseModel):
    task: str = Field(description="Label for the task (e.g., PourWaterFromBottle)")
    PreMotionPhase: FullPhase
    InitiationPhase: InitialPhase
    ExecutionPhase: Phase
    InteractionPhase: Phase
    TerminationPhase: Phase
    PostMotionPhase: Phase

