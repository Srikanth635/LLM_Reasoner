from .attribute_enums import *
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class Properties(BaseModel):
    size: Optional[Size] = Field(None, description="Physical size of the entity")
    length: Optional[Length] = Field(None, description="Measurement along the longest dimension")
    width: Optional[Width] = Field(None, description="Measurement along the shorter horizontal dimension")
    height: Optional[Height] = Field(None, description="Measurement along the vertical dimension")
    volume: Optional[Volume] = Field(None, description="Amount of space occupied by the entity")
    shape: Optional[Shape] = Field(None, description="Geometric form or outline of the entity")
    symmetry: Optional[Symmetry] = Field(None, description="Type of symmetrical properties")
    color: Optional[Color] = Field(None, description="Visual color appearance")
    texture: Optional[Texture] = Field(None, description="Surface feel or tactile quality")
    pattern: Optional[Pattern] = Field(None, description="Visual design or arrangement on the surface")
    reflectance: Optional[Reflectance] = Field(None, description="How light reflects off the surface")
    transparency: Optional[Transparency] = Field(None, description="Degree to which light passes through")
    material: Optional[Material] = Field(None, description="Substance the entity is made of")
    weight: Optional[Weight] = Field(None, description="Heaviness or mass of the entity")
    density: Optional[Density] = Field(None, description="Compactness of matter within the entity")
    firmness: Optional[Firmness] = Field(None, description="Resistance to deformation or pressure")
    grip: Optional[Grip] = Field(None, description="Quality of surface for holding or grasping")
    balance: Optional[Balance] = Field(None, description="Distribution of weight within the entity")
    handle: Optional[Handle] = Field(None, description="Presence and type of gripping mechanism")
    blade: Optional[Blade] = Field(None, description="Presence and characteristics of cutting edge")
    edge: Optional[Edge] = Field(None, description="Characteristics of the outer boundary")
    point: Optional[Point] = Field(None, description="Sharp or blunt tip characteristics")
    corners: Optional[Corners] = Field(None, description="Angular intersection characteristics")
    skin: Optional[Skin] = Field(None, description="Outer covering or peel characteristics")
    cleanliness: Optional[Cleanliness] = Field(None, description="State of hygiene or dirt")
    condition: Optional[Condition] = Field(None, description="Current state or form of the entity")
    intactness: Optional[Intactness] = Field(None, description="Degree of wholeness or completeness")
    freshness: Optional[Freshness] = Field(None, description="Quality related to newness or spoilage")
    ripeness: Optional[Ripeness] = Field(None, description="Stage of maturation for organic items")
    dirt: Optional[Dirt] = Field(None, description="Amount of accumulated soil or grime")
    count: Optional[Count] = Field(None, description="Quantity or number of items")
    orientation: Optional[Orientation] = Field(None, description="Spatial positioning or alignment")
    position: Optional[Position] = Field(None, description="Location relative to other objects")
    odor: Optional[Odor] = Field(None, description="Smell or scent characteristics")
    # Additional custom properties (for things like tool type, fingers, capability, state, surface, area, etc.)
    type: Optional[str] = Field(None, description="Specific type or category classification")
    fingers: Optional[str] = Field(None, description="Number or type of grasping appendages")
    capability: Optional[str] = Field(None, description="Functional ability or action potential")
    state: Optional[str] = Field(None, description="Current operational or physical state")
    surface: Optional[str] = Field(None, description="Type of surface material or finish")
    area: Optional[str] = Field(None, description="Spatial region or zone designation")

    # class Config:
    #     extra = "allow"  # Allow any additional properties not defined above


class Action(BaseModel):
    type: ActionTypes = Field(description="The type of action being performed (e.g., Opening, PickingUp, Placing, Pouring etc.,)")


class Object(BaseModel):
    type: str = Field(description="Classification of the object (e.g., PhysicalArtifact, Container)")
    name: str = Field(description="Specific name or identifier of the object")
    properties: Properties = Field(description="Physical and descriptive properties of the object")


class Tool(BaseModel):
    type: str = Field(description="Classification of the tool (e.g., Gripper, Hand, Utensil)")
    name: str = Field(description="Specific name or identifier of the tool")
    properties: Properties = Field(description="Physical and functional properties of the tool")


class Location(BaseModel):
    type: str = Field(description="Classification of the location (e.g., PhysicalPlace, Container, Surface)")
    name: str = Field(description="Specific name or identifier of the location")
    properties: Properties = Field(description="Physical and environmental properties of the location")


class CRAMActionDesignator(BaseModel):
    action: Action = Field(description="The action component specifying what is being done")
    object: Object = Field(description="The object component specifying what is being acted upon")
    tool: Optional[Tool] = Field(description="The tool component specifying what is performing the action")
    location: Optional[Location] = Field(description="The location component specifying where the action takes place")

    # def to_json(self, indent: int = 2) -> str:
    #     """Convert to JSON string with proper formatting"""
    #     return self.model_dump_json(indent=indent)
    #
    # def to_dict(self) -> Dict[str, Any]:
    #     """Convert to dictionary"""
    #     return self.model_dump()

