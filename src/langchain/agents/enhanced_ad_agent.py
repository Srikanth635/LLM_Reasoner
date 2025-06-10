import os.path

from pydantic import BaseModel
import json
from src.langchain.llm_configuration import *
from typing import TypedDict, Union, Optional
from langgraph.graph import StateGraph,START,END, MessagesState
from langchain_core.prompts import ChatPromptTemplate
import re
from pydantic import BaseModel
from pathlib import Path
from src.resources.pycram.cram_models import *

action_classes = [Adding,
    Arranging,
    Baking,
    Shutting,
    Cooking,
    Cooling,
    Cutting,
    Evaluating,
    Filling,
    Flavouring,
    Flipping,
    PickingUp,
    Lifting,
    Mixing,
    Neutralizing,
    Opening,
    OpeningADoor,
    OperatingATap,
    Pipetting,
    Pouring,
    Preheating,
    Pressing,
    Pulling,
    Putting,
    Removing,
    Rolling,
    Serving,
    Shaking,
    Spooning,
    Spreading,
    Sprinkling,
    Starting,
    Stopping,
    Stirring,
    Storing,
    Taking,
    Turning,
    TurningOnElectricalDevice,
    Unscrewing,
    UsingMeasuringCup,
    UsingSpiceJar,
    Waiting]

def read_json_from_file(filename):
    """Read JSON data from a file"""
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
json_path = os.path.join(src_dir, "resources", "pycram", "cram_action_cores.json")

json_data = read_json_from_file(json_path)

class ADState(TypedDict):
    instruction : str
    action_type : str
    action_core : str
    object_type : str
    tool_type : str
    location_type : str
    designator : str

class ADAction(BaseModel):
    type : dict

class ADObject(BaseModel):
    type : str
    name : str
    properties : dict[str,str]

class ADTool(BaseModel):
    type : Union[str, None]
    name : Union[str, None]
    properties : Union[dict[str,str],None]

class ADLocation(BaseModel):
    type : str
    name : str
    properties : dict[str,str]

class ADDesignator(BaseModel):
    action : ADAction
    object : Optional[ADObject] = None
    tool : Optional[ADTool] = None
    location : Optional[ADLocation] = None


mod1_template = """
    You are an expert in robotic task interpretation. Your task is to analyze a natural language instruction and identify 
    the **core action** being performed by the robot.

    Your output must be a single JSON object with the canonical action type selected from the list of **supported action cores**.


    Output a single JSON object like:
    
      "action": 
        "type": "<CanonicalActionType>"
      
    
    ### Allowed Action Types (Action Cores):
    Choose only one of the following canonical action types:
    
    ["Adding", "Arranging", "Baking", "Shutting", "Cooking", "Cooling", "Cutting", "Evaluating", "Filling", "Flavouring", "
    Flipping", "Lifting", "Mixing", "Neutralizing", "Opening", "OpeningADoor", "OperatingATap", "Pipetting", "PickingUp",
    "Pouring", "Preheating", "Pressing", "Pulling", "Placing", "Removing", "Rolling", "Serving", "Shaking", "Spooning", 
    "Spreading", "Sprinkling", "Starting", "Stopping", "Stirring", "Storing", "Taking", "Turning", "TurningOnElectricalDevice", 
    "Unscrewing", "UsingMeasuringCup", "UsingSpiceJar", "Waiting"]
    
    Examples:
    - "Pick up the cup" → "PickingUp"
    - "Place the spoon in the sink" → "Placing"
    - "Grab the plate" → "PickingUp"
    
    Example Output:
    "action": 
        "type": "PickingUp"
    
    NOTE: Do not include any explanations or feedback—strictly return only the corresponding JSON object.
    
    Instruction: {instruction}
    
"""

mod2_template = """
    You are a semantic reasoning agent specialized in extracting and describing the **main object** involved in a robotic action.

    Given a natural language instruction, identify the **primary physical object** that is being acted upon. This object 
    is the **target of the action** (e.g., picked up, moved, placed). It is not the tool, nor the location, but the **thing the robot is interacting with directly**.
    
    ### Rules:
    - The object is the noun **receiving the action** (e.g., spoon in “pick up the spoon”).
    - Only extract the object if it is explicitly mentioned.
    - Use the standardized list of properties and values below to semantically enrich the object.
    
    ### 📦 PROPERTY VALUE REFERENCE
    
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
    
    Output format:
    ```json

      "object":
        "type": "PhysicalArtifact",
        "name": "<ObjectName>",
        "properties": 
          // only use above parameters and values
        
      
    
    NOTE: Do not include any explanations or feedback—strictly return only the corresponding JSON object.
    And be very particular on the output format, a json dict with keys type, name and properties are needed with their corresponding values and
    for the properties key-value pairs are needed for relevant and allowed property values pairs. Any sort Explanations are strictly not required.
    
    Instruction: {instruction}

"""

mod3_template = """
    You are a semantic reasoning agent specialized in identifying and describing **tools** used in robotic tasks.

    Given a natural language instruction, identify whether a **tool** is mentioned. A tool is **not the object being acted
    on**, but something that is used to perform the action (e.g., a knife to cut, a spatula to flip, a robotic gripper to grasp).
    
    ### Rules:
    - If no tool is mentioned explicitly, return `null`.
    - Do **not confuse the object of the action** (e.g., spoon) with a tool (e.g., knife).
    - If a tool is present, return its name, type, and semantically enriched properties using the list below.


    ### 📦 PROPERTY VALUE REFERENCE
    
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
    
    If no tool is mentioned, return `null`. Otherwise, output:
    
    ```json
    
      "tool": 
        "type": "<ToolType>",
        "name": "<ToolName>",
        "properties": 
          // only from the allowed list
        
      
    
    NOTE: Do not include any explanations or feedback—strictly return only the corresponding JSON object.
    
    Instruction: {instruction}


"""

mod4_template = """
    You are a semantic reasoning agent specialized in extracting and describing **locations or targets** in robotic instructions.

    Given a natural language instruction, identify if there is a **location or spatial context** involved in the action. 
    This is where the action **starts or ends** (e.g., “from the shelf”, “into the drawer”).
    
    ### Rules:
    - The location is a **place** associated with the object’s position or movement, not the object or tool itself.
    - Only extract the location if it is **explicitly stated** or clearly implied (e.g., “on the shelf”, “into the dishwasher”).
    - Enrich the location using only the property values from the list below.

    ### 📦 PROPERTY VALUE REFERENCE
    
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
    
    If no location or target is present, return `null`. Otherwise, output:
    
    ```json
    
      "location": 
        "type": "<LocationType>",
        "name": "<LocationName>",
        "properties": 
          // only from the allowed list
        
    
    NOTE: Do not include any explanations or feedback—strictly return only the corresponding JSON object.
    
    Instruction: {instruction}


"""

mod5_template = """
    You are a CRAM task assembler. Given separate JSON blocks representing the action, object, tool (optional), and l
    ocation/target (optional), merge them into a single, valid JSON object following CRAM action designator format.

    Omit `tool` or `location` fields if they are null.
    
    Inputs:
    - Action: {action_block}
    - Object: {object_block}
    - Tool: {tool_block}
    - Location: {location_block}
    
    Output:
    <final JSON>

    NOTE: Do not include any explanations or feedback—strictly return only the corresponding JSON object.
"""

uni_template = """
    You are a CRAM task planner. Given an natural language instruction, action core and the required attributes to symbolically represent
    the action, you should fill in the values of those attributes from the provided context.
    
    Like for example: Pick up the cup from the table
    action core is PickingUp and 
    required attributes are "obj_to_be_grabbed", "action_verb", "location"  
    
    you should populate the attributes like obj_to_be_grabbed=cup action_verb=PickingUp, location=table.
    
    Given,
    instruction : {instruction}
    action_core : {action_core}
    required_attributes : {required_attributes}
    
"""

cram_template = """
    You are a CRAM task planner. Given action core attributes information and syntax cram plan for that action core,
    you'll are able to generate an instantiated cram plan with the information from the action core attributes information.
    
    Given,
    action_core_attributes : {action_core}
    syntax_cram_plan = {syntax_cram_plan}
    
"""

def think_remover(res : str):
    if re.search(r"<think>.*?</think>", res, flags=re.DOTALL):
        cleaned_res = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip()
    else:
        cleaned_res = res.strip()

    return cleaned_res


def action_node(state: ADState):
    print("INSIDE ACTION NODE")

    instruction = state["instruction"]

    mod1_prompt = ChatPromptTemplate.from_template(mod1_template)

    chain = mod1_prompt | ollama_llm.with_structured_output(ADAction, method="json_schema")

    response = chain.invoke({'instruction': instruction})

    dix = response.type
    try:
        core = list(dix.values())
        print(core)
        # print(type(dix), dix['action'])
        # print(json_data[dix['action']])
        # clean_response = think_remover(response.content)
        return {'action_type': response.model_dump_json(indent=2), 'action_core': core[0]}

    except:
        raise Exception("Parsing Error , Try Rerunning", print(response))

def object_node(state: ADState):
    print("INSIDE OBJECT NODE")

    instruction = state["instruction"]

    mod2_prompt = ChatPromptTemplate.from_template(mod2_template)

    chain = mod2_prompt | ollama_llm.with_structured_output(ADObject, method="json_schema")

    response = chain.invoke({'instruction': instruction})

    # clean_response = think_remover(response.content)

    return {'object_type': response.model_dump_json(indent=2)}

def tool_node(state: ADState):
    print("INSIDE TOOL NODE")

    instruction = state["instruction"]

    mod3_prompt = ChatPromptTemplate.from_template(mod3_template)

    chain = mod3_prompt | ollama_llm.with_structured_output(ADTool, method="json_schema")

    response = chain.invoke({'instruction': instruction})

    # clean_response = think_remover(response.content)

    return {'tool_type': response.model_dump_json(indent=2)}

def location_node(state: ADState):
    print("INSIDE LOCATION NODE")

    instruction = state["instruction"]

    mod4_prompt = ChatPromptTemplate.from_template(mod4_template)

    chain = mod4_prompt | ollama_llm.with_structured_output(ADLocation, method="json_schema")

    response = chain.invoke({'instruction': instruction})

    # clean_response = think_remover(response.content)

    return {'location_type': response.model_dump_json(indent=2)}

def action_designator_node(state:ADState):
    print("INSIDE ACTION DESIGNATOR NODE")

    action = state["action_type"]
    object = state["object_type"]
    tool = state["tool_type"]
    location = state["location_type"]

    mod5_prompt = ChatPromptTemplate.from_template(mod5_template)

    chain = mod5_prompt | ollama_llm.with_structured_output(ADDesignator, method="json_schema")

    response = chain.invoke({'action_block': action, 'object_block' : object,
                             'tool_block' : tool, 'location_block' : location})

    # clean_response = think_remover(response.content)

    return {'designator': response.model_dump_json(indent=2)}

def universal(state : ADState):
    print("INSIDE UNIVERSAL NODE")
    instruction = state['instruction']
    action_core = state['action_core']
    required_fields = json_data[action_core]['action_roles']
    cram_plan = json_data[action_core]['cram_plan']
    print("cram_plan", cram_plan)

    action_cls = next(
        (cls for cls in action_classes if cls.__name__ == action_core or
         getattr(cls, 'action_core', None) == action_core),
        None
    )
    if action_cls is None:
        raise ValueError(f"Unknown action_core: {action_core}")

    uni_prompt = ChatPromptTemplate.from_template(uni_template)
    cram_prompt = ChatPromptTemplate.from_template(cram_template)

    chain = uni_prompt | ollama_llm.with_structured_output(action_cls, method="json_schema")

    response = chain.invoke({'instruction': instruction, 'action_core': action_core, 'required_attributes': str(required_fields)})

    print(response)

    chain2 = cram_prompt | ollama_llm

    response2 = chain2.invoke({'action_core': response.model_dump_json(indent=2), 'syntax_cram_plan': cram_plan})

    print("CRAM PLAN ", response2.content)

    return {}


ad_builder = StateGraph(ADState)
ad_builder.add_node("action_node", action_node)
ad_builder.add_node("universal_node", universal)
# ad_builder.add_node("object_node", object_node)
# ad_builder.add_node("tool_node", tool_node)
# ad_builder.add_node("location_node", location_node)
# ad_builder.add_node("action_designator_node", action_designator_node)

# ad_builder.add_edge(START, "action_node")
# ad_builder.add_edge(START, "object_node")
# ad_builder.add_edge(START, "tool_node")
# ad_builder.add_edge(START, "location_node")

# ad_builder.add_edge("action_node", "action_designator_node")
# ad_builder.add_edge("object_node", "action_designator_node")
# ad_builder.add_edge("tool_node", "action_designator_node")
# ad_builder.add_edge("location_node", "action_designator_node")

# ad_builder.add_edge("action_designator_node", END)


ad_builder.add_edge(START, "action_node")
ad_builder.add_edge("action_node","universal_node")
ad_builder.add_edge("universal_node",END)

ad_graph = ad_builder.compile()

if __name__ == "__main__":
    print()

    # ress = ad_graph.invoke({"instruction": "pour the water from the mug into sink"})

    print(ad_graph.invoke({"instruction": "pour the water from the mug into sink"}))

    # print(json_path)
    # print(json_data['Pouring'])
    # print(ress['designator'])