from langgraph.graph import StateGraph,START,END, MessagesState
from langgraph.graph.message import add_messages
from typing import Annotated, List

class StateModel(MessagesState):
    instruction : str
    action_type : str
    action_core : str
    action_core_attributes : str
    enriched_action_core_attributes : str
    cram_plan_response : str
    premotion_phase : str
    phaser : str
    flanagan : str
    framenet_model : str
    context: str

class ModelsStateInternal(MessagesState):
    instruction : str
    action_type : str
    action_core : str
    action_core_attributes : str
    enriched_action_core_attributes : str
    cram_plan_response : str
    premotion_phase : str
    phaser : str
    flanagan : str
    framenet_model : str
    context: str

class ADStateInternal(MessagesState):
    instruction : str
    action_type : str
    action_core : str
    action_core_attributes : str
    enriched_action_core_attributes : str
    cram_plan_response : str
    context: str

class PyCramStateInternal(MessagesState):
    instruction : str
    action_names: List[str]
    action_models: Annotated[list, add_messages]