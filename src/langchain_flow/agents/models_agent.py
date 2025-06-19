from langgraph.checkpoint.memory import MemorySaver
from src.langchain_flow.agents.framenet_agent import framenet_node_pal_custom
from src.langchain_flow.agents.flanagan_agent import flanagan_node
from src.langchain_flow.agents.enhanced_ad_agent import *
from src.langchain_flow.state_graph import *

models_memory = MemorySaver()



# Aggregator Node
def aggregator_node(state: MessagesState):
    # messages = [
    #                {"role": "system", "content": framenet_system_prompt},
    #            ] + state["messages"]
    # print("Aggregator Node Messages", state["messages"])
    return {"messages" : "Message from Aggregator Node"}

model_builder = StateGraph(ModelsStateInternal)

model_builder.add_node("framenet_reasoner", framenet_node_pal_custom)
model_builder.add_node("flanagan_reasoner",flanagan_node)
model_builder.add_node("aggregator", aggregator_node)

model_builder.add_edge(START, "framenet_reasoner")
model_builder.add_edge(START, "flanagan_reasoner")

model_builder.add_edge("flanagan_reasoner", "aggregator")
model_builder.add_edge("framenet_reasoner", "aggregator")
model_builder.add_edge("aggregator", END)

model_graph = model_builder.compile(checkpointer=models_memory)


def models_node(state: StateModel):
    instruction = state['instruction']
    action_type = state['action_type']
    action_core = state['action_core']
    action_core_attributes = state['action_core_attributes']
    enriched_action_core_attributes = state['enriched_action_core_attributes']
    cram_plan_response = state['cram_plan_response']


    final_models_state = model_graph.invoke({'instruction' : instruction, 'action_type' : action_type, 'action_core' : action_core,
                        'action_core_attributes' : action_core_attributes, 'enriched_action_core_attributes' : enriched_action_core_attributes,
                        'cram_plan_response' : cram_plan_response})

    framenet_model = final_models_state['framenet_model']
    premotion_phase = final_models_state['premotion_phase']
    phaser = final_models_state['phaser']
    flanagan = final_models_state['flanagan']


    return {'framenet_model' : framenet_model, 'premotion_phase' : premotion_phase, 'phaser' : phaser, 'flanagan' : flanagan}

if __name__ == "__main__":
    print()