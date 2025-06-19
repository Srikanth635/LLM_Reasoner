from langgraph.graph import StateGraph,START,END, MessagesState

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