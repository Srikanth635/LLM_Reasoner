from dotenv.variables import Literal
from langgraph.graph import MessagesState
from src.langchain.create_agents import *
from src.langchain.llm_configuration import *
from src.resources.ad_model.ad_modes import CRAMActionDesignator
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent, InjectedState
from src.langchain.create_agents import *
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
import re
from langchain_core.messages import HumanMessage, ToolMessage
import json
from langchain_core.runnables import RunnableConfig

agent_sys_prompt = """
    You are a robotic reasoning agent responsible for converting natural language instructions into structured CRAM-style action designators.
    
    Your task is to interpret user-provided robot task instructions and produce semantic, machine-readable JSON outputs that describe what the robot must do in the form of a CRAM action designator. To achieve this, you must use a designated tool that returns the complete designator structure.
    
    ---
    
    TOOL ACCESS
    
    You have access to one tool:
    
    `model_selector(instruction: str) -> dict`
    
    - This tool analyzes the instruction and returns a fully structured CRAM-style action designator as a Python dictionary (JSON-like).
    - The designator includes fields for action, object, tool, and location—each with nested properties.
    - You must invoke this tool for every instruction and return its response as-is.
    
    This tool is the only mechanism for generating action designators. Do not write them manually.
    
    ---
    
    OBJECTIVE
    
    Your job is to:
    
    1. Accept a user instruction.
    2. Invoke the `model_selector` tool with that instruction.
    3. Return the resulting action designator exactly as provided by the tool.
    
    The action designator must include:
    
    - action: type of robot action (e.g., PickingUp, Placing)
    - object: physical item involved, with properties like material, size, color, etc.
    - tool: actuator used, typically a gripper, with technical properties
    - location: where the action occurs, including surface, area, and environmental attributes
    
    ---
    
    REASONING INSTRUCTIONS
    
    - Always invoke the tool to obtain the action designator. Do not attempt to infer or fabricate the designator yourself.
    - Trust the tool's structure and return it directly.
    - If the tool output is incomplete or needs inference, make safe and reasonable assumptions to ensure completeness.
    - Output must always be valid JSON.
    
    ---
    
    OUTPUT FORMAT
    
    The tool returns a structure like:
    
    {
      "action": {
        "type": "PickingUp"
      },
      "object": {
        "type": "PhysicalArtifact",
        "name": "Bottle",
        "properties": {
          "material": "plastic",
          "color": "red",
          "size": "medium",
          "shape": "cylindrical",
          "cleanliness": "clean",
          "weight": "light"
        }
      },
      "tool": {
        "type": "Gripper",
        "name": "RobotGripper",
        "properties": {
          "type": "parallel-jaw",
          "fingers": "two",
          "material": "rubberized",
          "capability": "grasping",
          "state": "open"
        }
      },
      "location": {
        "type": "PhysicalPlace",
        "name": "FloorNearSink",
        "properties": {
          "surface": "tile",
          "area": "next-to-sink",
          "height": "zero",
          "material": "ceramic",
          "color": "white"
        }
      }
    }
    
    ---
    
    EXAMPLE:
    
    Input:  
    "Pick up the red bottle from the floor next to the sink."
    
    Tool Output:  
    (The complete JSON as above)
    
    Final Response:  
    Return the output from the tool directly—no changes, no reformats.
    
    ---
    
    You are a facilitator between user instructions and the structured designator generator. Always route through the tool and return its output faithfully.
"""

action_designators = []

ad_llm = ollama_llm.with_structured_output(CRAMActionDesignator, method="json_schema")


user_prompt_template = """

    You are an intelligent semantic reasoning agent tasked with interpreting freeform natural language instructions and 
    converting them into machine-readable CRAM-style action designators—structured JSON objects that clearly represent 
    the intended robotic action, the object involved, the tool used, and the location of the task, all enriched with 
    detailed and contextually accurate properties for each element.

    ---

    ### STRUCTURE OVERVIEW

    Your output must be a JSON object with the following fields:

    1. **action**: What is being done.
    2. **object**: The physical item involved.
    3. **tool**: The actuator or mechanism performing the task.
    4. **location**: The physical space where the action occurs.

    Each section must include nested `properties` containing semantic details. Use the format shown in the example.

    ---

    ### 📦 PROPERTY VALUE REFERENCE

    Use these standardized property values where applicable:

    - size: small, medium, large, tiny, huge
    - length: short, medium, long
    - width: narrow, medium, wide
    - height: low, medium, high, tall
    - volume: low, medium, high
    - shape: round, oval, square, rectangular, cylindrical, conical, irregular, flat, spherical, cubical, hemispherical
    - symmetry: radial, bilateral, asymmetric, none
    - color: red, green, blue, yellow, orange, purple, brown, black, white, grey, clear
    - texture: smooth, rough, bumpy, fuzzy, prickly, slippery, sticky, grainy, layered, flaky, crisp, soft, hard, waxy, powdery
    - pattern: solid, striped, spotted, marbled, checked, floral, graphic, none
    - reflectance: glossy, matte, shiny, dull
    - transparency: opaque, translucent, transparent
    - material: metal, plastic, ceramic, glass, wood, fabric, paper, rubber, silicone, stone, organic
    - weight: light, medium, heavy
    - density: lowdensity, mediumdensity, highdensity, dense
    - firmness: soft, medium, firm, hard, rigid, squishy, brittle
    - grip: smooth, textured, easygrip, slippery, secure
    - balance: balanced, bladeheavy, handleheavy
    - handle: present, none, single, double, loop, straight
    - blade: present, none, straight, serrated, curved, short, long, sharp, dull
    - edge: present, none, sharp, dull, straight, curved, beveled, rounded
    - point: present, none, sharp, rounded, blunt
    - corners: present, none, sharp, rounded
    - skin: present, none, thin, thick, smooth, rough, peeled
    - cleanliness: clean, dirty, washed, sticky, greasy
    - condition: whole, cut, sliced, diced, chopped, peeled, bruised, broken, cracked, bent, damaged, good, bad
    - intactness: intact, broken, damaged, partial
    - freshness: fresh, stale, wilting, expired
    - ripeness: unripe, ripe, overripe
    - dirt: none, some, heavy
    - count: single, multiple, few, many
    - orientation: upright, sideways, inverted, angled
    - position: on, in, under, near, far, left, right, front, back, center, edge
    - odor: none, mild, strong, sweet, sour, pungent, aromatic, burnt, spicy

    ---

    ### OUTPUT EXAMPLE 1

    Input:  
    "Pick up the dropped spoon from the floor near the table."

    Output:
    
      "action": 
        "type": "PickingUp"
      ,
      "object": 
        "type": "PhysicalArtifact",
        "name": "Spoon",
        "properties": 
          "material": "stainless-steel",
          "color": "silver",
          "size": "small",
          "shape": "spoon-shaped",
          "texture": "smooth",
          "cleanliness": "dirty",
          "weight": "light",
          "condition": "dropped"
      ,
      "tool": 
        "type": "Gripper",
        "name": "RobotGripper",
        "properties": 
          "type": "parallel-jaw",
          "fingers": "two",
          "material": "rubberized",
          "capability": "grasping",
          "state": "open"
      ,
      "location": 
        "type": "PhysicalPlace",
        "name": "FloorNearTable",
        "properties": 
          "surface": "tile",
          "area": "under-table",
          "height": "zero",
          "color": "white",
          "material": "ceramic",
          "cleanliness": "dirty"
    
    OUTPUT EXAMPLE 2:
    
    Input:  
    "Grab the spatula from the utensil holder."
    
    Output:
    
        "action": 
          "type": "PickingUp"
        ,
        "object": 
          "type": "PhysicalArtifact",
          "name": "Spatula",
          "properties": 
            "material": "silicone",
            "color": "black",
            "size": "long",
            "shape": "flat",
            "handle": "present",
            "cleanliness": "clean",
            "weight": "light",
            "flexibility": "flexible"        
        ,
        "tool": 
          "type": "Gripper",
          "name": "RobotGripper",
          "properties": 
            "type": "parallel-jaw",
            "fingers": "two",
            "material": "rubberized",
            "capability": "grasping",
            "state": "open"       
        ,
        "location": 
          "type": "PhysicalPlace",
          "name": "UtensilHolder",
          "properties": 
            "type": "countertop",
            "material": "ceramic",
            "color": "grey",
            "contents": "utensils",
            "state": "full",
            "position": "beside-stove",
            "height": "medium"

    OUTPUT EXAMPLE 3:
    
    Input:  
    "Place the glass measuring cup inside the dishwasher."
    
    Output:
    
        "action": 
          "type": "Placing"
        ,
        "object": 
          "type": "PhysicalArtifact",
          "name": "MeasuringCup",
          "properties": 
            "material": "glass",
            "color": "clear",
            "size": "medium",
            "shape": "cylindrical",
            "weight": "medium",
            "transparency": "transparent",
            "condition": "intact"    
        ,
        "target": 
          "type": "PhysicalPlace",
          "name": "Dishwasher",
          "properties": 
            "type": "appliance",
            "material": "metal",
            "position": "under-counter",
            "doorState": "open"
        ,
        "tool": 
          "type": "Gripper",
          "name": "RobotGripper",
          "properties": 
            "type": "parallel-jaw",
            "fingers": "two",
            "material": "rubberized",
            "capability": "grasping",
            "state": "closed"
    
    ---

    ### 🧠 REASONING INSTRUCTIONS

    - Use all relevant properties for each entity (object, tool, location).
    - Use you semantic knowledge to infer missing details of entities like geometric properties like dimensions, appearance 
        properties like color, texture, pattern etc., physical properties like weight, firmness, density, etc., with
        most probable values suited for the entity.
    - Infer details using commonsense (e.g., dropped spoons are dirty and light) and always assume ideal conditions for entities unless specified
        like clea
    - Never omit a field if it can be reasonably deduced.
    - Be realistic, grounded, and consistent with physical world knowledge.
    - Output must be valid JSON. Maintain camelCase for all property keys.

    Generate cram action designator for the given instruction {instruction}
    
    """


user_prompt = ChatPromptTemplate.from_template(user_prompt_template)
# ad_agent = create_react_agent(model=ad_llm, tools=[], prompt=agent_sys_prompt)

@tool(description="CRAM action designator generator from natural language instruction",
      return_direct=True)
def entity_attribute_finder(instruction: str):
    """
    Generates CRAM Action Designator with entity descriptions with their attributes and values
    :param instruction: input user natural language instruction
    :param state: Graph state
    :param config: Runnable Configuration
    :param tool_call_id: Tool call id
    :return: CRAM action designator in json string format
    """
    print("INSIDE ENTITY ATTRIBUTE FINDER")
    print("%" * 10)
    # print("Current Graph State : ", state["messages"])
    # print("config user id : ",config["configurable"].get("user_id"))
    # print("tool call id : ", tool_call_id)
    print("%"*10)
    chain =  user_prompt | ad_llm
    response = chain.invoke({"instruction": instruction})
    # response_content = response['messages'][-1].content
    cleaned_response = re.sub(r'<think>.*?</think>', '', str(response), flags=re.DOTALL).strip()
    action_designators.append(cleaned_response)
    # print("Cleaned Response : ", cleaned_response)
    json_response = response.model_dump_json(indent=2, by_alias=True)
    return json_response
    # return Command(update={
    #     "messages" : [ToolMessage(content=json_response, name="entity_attribute_finder", tool_call_id=tool_call_id)]
    # },
    # goto='ad_agent')


ad_agent = create_agent(llm=ollama_llm, tools=[entity_attribute_finder], agent_sys_prompt=agent_sys_prompt)
# ad_agent = create_react_agent(model=ollama_llm, tools=[entity_attribute_finder], prompt=agent_sys_prompt, checkpointer=False)

def ad_agent_node_pal(state: MessagesState):

    # instruction = state["messages"][-1]
    #
    # ad_chain = user_prompt | ad_agent
    #
    # result = ad_chain.invoke({"instruction": instruction.content})
    #
    # response = result['messages'][-1].content
    # cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    #
    # print(f"cleaned response {cleaned_response}")
    result = ad_agent.invoke(state)
    # print("ad agent response : ", result["messages"][-1])


    return {
        "messages" : [HumanMessage(content=result["messages"][-1].content, name="ad_agent_node_pal")]
        # "messages": result["messages"][-1]
        # "messages": cleaned_response.strip()
    }

if __name__ == "__main__":
    # import re

    ad_chain = user_prompt | ad_agent
    # for chunk in ad_chain.stream({"instruction": "Pick up the dropped spoon from the floor near the table."}, stream_mode="values"):
    #     response = chunk['agent']['messages'][-1].content
    #     cleaned_text = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    #     print(cleaned_text.strip())
    #     print("------")
    # print(ad_chain.invoke({"instruction": "Pick up the dropped spoon from the floor near the table."}))
    # framenet_node_pal({"messages": [HumanMessage(content="Pick up the black spoon from the brown shelf")]})

    print(ad_chain.invoke({"instruction": "Pick up the black spoon from the brown shelf"}))