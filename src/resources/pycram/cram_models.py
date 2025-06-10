from pydantic import BaseModel
from typing import Optional, List

class Adding(BaseModel):
    """A Pydantic model for the 'Adding' action."""
    theme: str
    goal: str
    action_verb: str
    unit: Optional[str] = None
    amount: Optional[float] = None

class Arranging(BaseModel):
    """A Pydantic model for the 'Arranging' action."""
    action_verb: str
    obj_to_be_arranged: str

class Baking(BaseModel):
    """A Pydantic model for the 'Baking' action."""
    theme: str
    action_verb: str

class Shutting(BaseModel):
    """A Pydantic model for the 'Shutting' action."""
    obj_to_be_shut: str
    action_verb: str
    utensil: Optional[str] = None

class Cooking(BaseModel):
    """A Pydantic model for the 'Cooking' action."""
    obj_to_be_cooked: str
    action_verb: str

class Cooling(BaseModel):
    """A Pydantic model for the 'Cooling' action."""
    action_verb: str
    amount: Optional[float] = None
    location: Optional[str] = None
    obj_to_be_cooled: Optional[str] = None
    unit: Optional[str] = None

class Cutting(BaseModel):
    """A Pydantic model for the 'Cutting' action."""
    obj_to_be_cut: str
    utensil: str
    action_verb: str
    amount: Optional[float] = None
    unit: Optional[str] = None

class Evaluating(BaseModel):
    """A Pydantic model for the 'Evaluating' action."""
    obj_to_be_evaluated: str
    attribute: str
    action_verb: str

class Filling(BaseModel):
    """A Pydantic model for the 'Filling' action."""
    stuff: str
    goal: str
    action_verb: str

class Flavouring(BaseModel):
    """A Pydantic model for the 'Flavouring' action."""
    spice: str
    goal: str
    action_verb: str

class Flipping(BaseModel):
    """A Pydantic model for the 'Flipping' action."""
    obj_to_be_flipped: str
    action_verb: str
    utensil: Optional[str] = None

class PickingUp(BaseModel):
    """A Pydantic model for the 'PickingUp' action."""
    obj_to_be_grabbed: str
    action_verb: str
    location: Optional[str] = None

class Lifting(BaseModel):
    """A Pydantic model for the 'Lifting' action."""
    obj_to_be_lifted: str
    action_verb: str

class Mixing(BaseModel):
    """A Pydantic model for the 'Mixing' action."""
    content: List[str]
    action_verb: str

class Neutralizing(BaseModel):
    """A Pydantic model for the 'Neutralizing' action."""
    neutralizee: str
    neutralizer: str
    action_verb: str
    amount: Optional[float] = None
    unit: Optional[str] = None

class Opening(BaseModel):
    """A Pydantic model for the 'Opening' action."""
    obj_to_be_opened: str
    action_verb: str

class OpeningADoor(BaseModel):
    """A Pydantic model for the 'OpeningADoor' action."""
    obj_to_be_opened: str
    action_verb: str

class OperatingATap(BaseModel):
    """A Pydantic model for the 'OperatingATap' action."""
    liquid: str
    goal: str
    action_verb: str
    amount: Optional[float] = None
    unit: Optional[str] = None

class Pipetting(BaseModel):
    """A Pydantic model for the 'Pipetting' action."""
    content: str
    goal: str
    action_verb: str
    amount: Optional[float] = None
    unit: Optional[str] = None

class Pouring(BaseModel):
    """A Pydantic model for the 'Pouring' action."""
    stuff: str
    goal: str
    action_verb: str
    unit: Optional[str] = None
    amount: Optional[float] = None

class Preheating(BaseModel):
    """A Pydantic model for the 'Preheating' action."""
    obj_to_be_heated: str
    action_verb: str
    temperature_unit: Optional[str] = None
    temperature_setting: Optional[float] = None

class Pressing(BaseModel):
    """A Pydantic model for the 'Pressing' action."""
    obj_to_be_pressed: str
    action_verb: str
    location: Optional[str] = None

class Pulling(BaseModel):
    """A Pydantic model for the 'Pulling' action."""
    obj_to_be_pulled: str
    action_verb: str

class Putting(BaseModel):
    """A Pydantic model for the 'Putting' action."""
    obj_to_be_put: str
    action_verb: str
    location: Optional[str] = None

class Removing(BaseModel):
    """A Pydantic model for the 'Removing' action."""
    action_verb: str
    location: Optional[str] = None
    obj_to_be_removed: Optional[str] = None

class Rolling(BaseModel):
    """A Pydantic model for the 'Rolling' action."""
    theme: str
    action_verb: str

class Serving(BaseModel):
    """A Pydantic model for the 'Serving' action."""
    theme: str
    action_verb: str

class Shaking(BaseModel):
    """A Pydantic model for the 'Shaking' action."""
    obj_to_be_shaken: str
    action_verb: str
    unit: Optional[str] = None
    amount: Optional[float] = None

class Spooning(BaseModel):
    """A Pydantic model for the 'Spooning' action."""
    substance: str
    goal: str
    action_verb: str

class Spreading(BaseModel):
    """A Pydantic model for the 'Spreading' action."""
    substance: str
    goal: str
    action_verb: str

class Sprinkling(BaseModel):
    """A Pydantic model for the 'Sprinkling' action."""
    substance: str
    goal: str
    action_verb: str

class Starting(BaseModel):
    """A Pydantic model for the 'Starting' action."""
    obj_to_be_started: str
    action_verb: str

class Stopping(BaseModel):
    """A Pydantic model for the 'Stopping' action."""
    obj_to_be_stopped: str
    action_verb: str

class Stirring(BaseModel):
    """A Pydantic model for the 'Stirring' action."""
    action_verb: str
    content: List[str]

class Storing(BaseModel):
    """A Pydantic model for the 'Storing' action."""
    obj_to_be_stored: str
    action_verb: str
    location: Optional[str] = None

class Taking(BaseModel):
    """A Pydantic model for the 'Taking' action."""
    obj_to_be_taken: str
    action_verb: str
    location: Optional[str] = None

class Turning(BaseModel):
    """A Pydantic model for the 'Turning' action."""
    obj_to_be_turned: str
    action_verb: str

class TurningOnElectricalDevice(BaseModel):
    """A Pydantic model for the 'TurningOnElectricalDevice' action."""
    device: str
    action_verb: str

class Unscrewing(BaseModel):
    """A Pydantic model for the 'Unscrewing' action."""
    obj_to_be_unscrewed: str
    action_verb: str

class UsingMeasuringCup(BaseModel):
    """A Pydantic model for the 'UsingMeasuringCup' action."""
    content: str
    goal: str
    action_verb: str
    amount: Optional[float] = None
    unit: Optional[str] = None

class UsingSpiceJar(BaseModel):
    """A Pydantic model for the 'UsingSpiceJar' action."""
    content: str
    goal: str
    action_verb: str

class Waiting(BaseModel):
    """A Pydantic model for the 'Waiting' action."""
    unit: Optional[str] = None
    action_verb: Optional[str] = None
    amount: Optional[float] = None