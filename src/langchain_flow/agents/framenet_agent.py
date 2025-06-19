from http.client import responses
from typing import Literal
from src.langchain_flow.create_agents import *
from src.langchain_flow.llm_configuration import *
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import Tool
from src.langchain_flow.state_graph import *
from langgraph.graph import MessagesState
from langgraph.types import Command
from langgraph.prebuilt import InjectedState, ToolNode

import os


###############################Structured Output FrameNet START#######################################################

from pydantic import BaseModel, Field
from typing import Optional


class CoreElements(BaseModel):
    agent: Optional[str] = Field(description="Who performs the action (e.g., robot, user)")
    theme_patient: Optional[str] = Field(description="What is acted on (e.g., object being cut or moved)")
    instrument: Optional[str] = Field(description="Tool or means used")
    source: Optional[str] = Field(description="Where the object comes from")
    goal: Optional[str] = Field(description="Where the object is placed or directed")
    result: Optional[str] = Field(description="Resulting state or transformation")


class PeripheralElements(BaseModel):
    location: Optional[str] = Field(description="General location of the action")
    manner: Optional[str] = Field(description="How the action is done (e.g., gently, quickly)")
    direction: Optional[str] = Field(description="Direction of movement or force")
    time: Optional[str] = Field(description="When the action occurs")
    quantity: Optional[str] = Field(description="Quantity or amount involved")
    portion: Optional[str] = Field(description="Part of object affected")


class FrameNetRepresentation(BaseModel):
    framenet: str = Field(description="High-level label of the semantic task (e.g., cutting, placing)")
    frame: str = Field(description="FrameNet frame name")
    lexical_unit: str = Field(alias="lexical-unit", description="Lexical unit that evokes the frame")
    core: CoreElements
    peripheral: PeripheralElements


###############################Structured Output FrameNet END#########################################################
framenet_answers = []

framenet_prompt_template = """
    You are an expert in Frame Semantics. For each instruction given, generate a FrameNet-style structured representation. Use a YAML-style output format (no curly braces), with abstract but context-relevant labels. Include peripheral elements when they apply.
    
    Use this schema:
    
    framenet: <semantic label>            # e.g., picking_up, cutting, placing
    frame: <FrameNet frame name>          # e.g., Getting, Cutting, Placing
    lexical-unit: <FrameNet LU>          # e.g., pick up.v, cut.v, place.v
    
    core:
      agent:                              # who performs the action (e.g., robot, user)
      theme/patient:                      # the object acted on
      instrument:                         # tool used (if any)
      source:                             # original location or source
      goal:                               # intended destination or target state
      result:                             # final state or outcome
    
    peripheral:
      location:                           # general location (e.g., kitchen, shelf)
      manner:                             # how the action is done (e.g., gently)
      direction:                          # motion direction (e.g., upward)
      time:                               # temporal context (e.g., during prep)
      quantity:                           # amount involved (e.g., one item)
      portion:                            # part involved (e.g., top half)
    
    ---
    
    Example 1:
    Instruction: Pick up the bottle from the sink
    
    framenet: picking_up
    frame: Getting
    lexical-unit: pick up.v
    core:
      agent: robot
      theme/patient: bottle
      instrument: robot gripper
      source: sink
      goal:
      result: robot has the bottle
    peripheral:
      location: kitchen workspace
      manner: gently
      direction: upward
      time: during cleanup phase
      quantity: one item
      portion:
    
    ---
    
    Example 2:
    Instruction: Cut the apple into slices with a knife
    
    framenet: cutting
    frame: Cutting
    lexical-unit: cut.v
    core:
      agent: robot
      theme/patient: apple
      instrument: knife
      source:
      goal:
      result: apple slices
    peripheral:
      location: food prep surface
      manner: precisely
      direction: downward
      time: during preparation
      quantity: single apple
      portion: whole
    
    ---
    
    Example 3:
    Instruction: Place the cup on the shelf
    
    framenet: placing
    frame: Placing
    lexical-unit: place.v
    core:
      agent: robot
      theme/patient: cup
      instrument:
      source:
      goal: shelf
      result: cup is on the shelf
    peripheral:
      location: storage area
      manner: carefully
      direction: upward
      time: after drying
      quantity: one item
      portion:
    
    ---
    
    Now, given the instruction below, produce a similar output:
    
    Instruction: {input_instruction}
"""
framenet_prompt = ChatPromptTemplate.from_template(framenet_prompt_template)

# llm_fn = ChatOpenAI(model="gpt-4o-mini")
# structured_llm_fn = llm.with_structured_output(FrameNetRepresentation)
structured_ollama_llm_fn = ollama_llm.with_structured_output(FrameNetRepresentation, method="json_schema")

# Agent Specific Tools
@tool(description="framenet representation of the input string with direct return tool output",
      return_direct=True)
def framenet_tool(state: Annotated[dict, InjectedState]):
    """
    Generate a FrameNet-style semantic representation for a given natural language instruction.

    This function invokes a prompt chain using a predefined FrameNet prompt template and a language model
    function (llm_fn) to extract frame semantics such as agent, patient, instrument, etc., from the input instruction.

    Args:
        a (str): A natural language instruction (e.g., "Pick up the bottle from the sink").

    Returns:
        dict: A simplified dictionary with extracted semantic roles (e.g., {'agent': 'robot', 'patient': 'apple'}).
    """
    print("INSIDE FRAMENET TOOL")
    instruction = state["instruction"]
    chain = framenet_prompt | structured_ollama_llm_fn
    response = chain.invoke({"input_instruction": instruction})
    json_response = response.model_dump_json(indent=2, by_alias=True)
    framenet_answers.append(json_response)
    return json_response

# Framenet Agent State
class FramenetState(TypedDict):
    framenet : Annotated[list[FrameNetRepresentation], add_messages, Field(description="Framenet Representations")]
    messages : Annotated[list, add_messages]


# Agent
# framenet_agent = create_agent(ollama_llm, [framenet_tool], agent_state_schema=FramenetState)
# framenet_agent = create_framenet_agent(ollama_llm, [framenet_tool], agent_state_schema=FramenetState)
# framenet_agent = create_framenet_agent(ollama_llm, [framenet_tool], agent_state_schema=FramenetState)


# Agent as Node
# def framenet_node(state: MessagesState) -> Command[Literal["supervisor"]]:
#     # messages = [
#     #                {"role": "system", "content": framenet_system_prompt},
#     #            ] + state["messages"]
#     result = framenet_agent.invoke(state)
#     return Command(
#         update={
#             "messages": [
#                 HumanMessage(content=result["messages"][-1].content, name="framenet")
#             ]
#         },
#         goto="supervisor",
#     )
#
# def framenet_node_pal(state: StateModel):
#     # messages = [
#     #                {"role": "system", "content": framenet_system_prompt},
#     #            ] + state["messages"]
#
#     # framenet_agent_output_parser = PydanticOutputParser(pydantic_object=FrameNetRepresentation)
#     # agent_chain = framenet_agent | framenet_agent_output_parser
#     # result = agent_chain.invoke(state)
#     result = framenet_agent.invoke(state)
#
#     return {'framenet_model': framenet_answers[-1]}
#
#     # return {
#     #     "messages": result["messages"][-1]
#     # }

def framenet_node_pal_custom(state: ModelsStateInternal):
    print("INSIDE FRAMENET NODE")
    instruction = state["instruction"]
    chain = framenet_prompt | structured_ollama_llm_fn
    response = chain.invoke({"input_instruction": instruction})
    json_response = response.model_dump_json(indent=2, by_alias=True)
    framenet_answers.append(json_response)
    return {'framenet_model': json_response}


if __name__ == "__main__":
    print("INSIDE MAIN")

    # test_framenet()
    # print(framenet_tool("pick up the bottle from the sink"))
    # print(fnr[0])
    # framenet_agent.invoke({'messages' : [HumanMessage(content='pick up the bottle from the sink')]})