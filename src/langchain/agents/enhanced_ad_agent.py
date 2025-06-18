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
from src.langchain.state_graph import StateModel

action_classes = [Peeling,Adding,
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
    Placing,
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
    Waiting,
    Holding]

# Utility Functions
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

def think_remover(res : str):
    if re.search(r"<think>.*?</think>", res, flags=re.DOTALL):
        cleaned_res = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip()
    else:
        cleaned_res = res.strip()

    return cleaned_res

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
json_path = os.path.join(src_dir, "resources", "pycram", "cram_action_cores.json")

json_data = read_json_from_file(json_path)

class ADState(TypedDict):
    instruction : str
    action_type : str
    action_core : str
    action_core_attributes : str
    enriched_action_core_attributes : str
    cram_plan_response : str
    # object_type : str
    # tool_type : str
    # location_type : str
    # designator : str

class ADAction(BaseModel):
    type : dict


# -------------------------------------- Normal Entity Attribute Finder ----------------------------------
# Base Models
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

# Prompt Templates
mod1_template = """
    You are a decisive Robotic Action Classifier. Your single, critical task is to analyze a natural language instruction and
    classify it into one of the predefined action types from the provided list. Your decision-making must be swift and accurate.

    ### Core Directives ###
    - Always Choose One: 
        You must select one action core from the Allowed Action Types list. The "type" field in your output JSON must never be empty or null.
    - Find the Best Fit: 
        Some instructions may be ambiguous or use verbs not directly on the list (e.g., "get", "fetch", "move"). In these cases,
        you are to choose the action core that is the most logical and closest semantic match. It is mandatory to make a reasonable
        classification rather than providing no answer.
    - Strict Output Format: 
        Your response must only be the JSON object. Do not include any other text, notes, or explanations.


    ### Output Format ###
    Your output must be a single JSON object in this exact format:
    
    JSON
    
    "action": "type": "CanonicalActionType"
      
    
    ### Allowed Action Types (Action Cores):
    Choose only one of the following canonical action types:
    
    ["Cutting","Lifting", "OpeningADoor", "OperatingATap", "Pipetting", "PickingUp",
    "Pouring", "Pressing", "Pulling", "Placing", "Removing", "Rolling", "Shaking", "Spooning",
    "Spreading", "Sprinkling", "Stirring", "Taking", "Turning", "TurningOnElectricalDevice",
    "Unscrewing"]
    
    Examples:
    
    - Instruction: "Pick up the cup"
    - Your Output: "action": "type": "PickingUp"
    
    - Instruction: "Place the spoon in the sink"  
    - Your Output: "action": "type": "Placing"
    
    - Instruction: "Get me the bottle from the top shelf"
        Reasoning (for your internal logic): "Get" is ambiguous, but the most logical primary action is to take the object from its location.
    - Your Output: "action": "type": "Taking"
    
    - Instruction: "put the oven on at 180c"
        Reasoning (for your internal logic): "put on" in this context means starting an electrical device.
    - Your Output: "action": "type": "TurningOnElectricalDevice"
    
    ---
    
    Now perform the task for the given instruction :
    
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

# Nodes
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


# -------------------------------------- CRAM Designators  ----------------------------------

# Prompt Templates
field_prompt_template = """
    You are a hyper-specialized AI agent. Your sole purpose is to parse a natural language instruction and populate a dictionary of its attributes based on a provided list.

    Your Mission:
    
    You will receive three inputs:
    
    Natural Language Instruction: The full user command.
    Action Core: The primary action being analyzed (e.g., "Pouring", "Cutting").
    List of Target Attributes: A list of specific action roles to find (e.g., ["stuff", "goal", "amount"]).
    
    Your job is to analyze the instruction and find the corresponding value for each attribute in the list.
    You must then return a single JSON object containing all the attribute-value pairs.
    
    OUTPUT RULES:
    - Your response MUST be a single, valid JSON object.
    - The keys of the JSON object must exactly match the attributes from the input List of Target Attributes.
    - If a value for a specific attribute cannot be found in the instruction, its value in the JSON must be null (the JSON null type, not the string "null").
    - Extract the most concise and accurate phrase or value for each attribute.
    - Your final output should ONLY be the JSON object, with no surrounding text, explanations, or markdown formatting like json.
    
    
    ### Examples ###
    Here are examples of how you must respond.
    
    Example 1:
    
    Natural Language Instruction: "Carefully pour the pancake batter onto the hot griddle."
    Action Core: Pouring
    List of Target Attributes: ["stuff", "goal", "action_verb", "unit", "amount"]
    Your Output:
    JSON
    
    
      "stuff": "pancake batter",
      "goal": "griddle",
      "action_verb": "pour",
      "unit": null,
      "amount": null
    
    
    Example 2:

    Natural Language Instruction: "Dice one large onion with a chef's knife."
    Action Core: Cutting
    List of Target Attributes: ["obj_to_be_cut", "action_verb", "utensil", "amount"]
    Your Output:
    JSON
    
    
      "obj_to_be_cut": "one large onion",
      "action_verb": "Dice",
      "utensil": "a chef's knife",
      "amount": "one"
    
    
    
    Now perform the task as mentioned for given,
    
    instruction : {instruction}
    action_core : {action_core}
    target_attributes : {target_attributes}
    
    
    /nothink
"""

field_props_prompt_template = """
    You are a sophisticated AI agent acting as a Semantic Enrichment Engine. Your primary function is to take a JSON object, 
    analyze its contents, and enrich it with relevant semantic metadata. You must infer properties and also extract them 
    when they are explicitly stated.

    Your Mission:
    
    You will receive a JSON object. Your task is to:
    
    - Iterate through each key-value pair in the JSON.
    - For each attribute with a non-null value, determine if it can be semantically described (e.g., an object, a location, 
    a substance). Attributes like simple action verbs are typically not enriched.
    - If an attribute is relevant, you will:
        - Add a new key to the JSON. This new key must follow the format: [attribute_name]_props. For example, for the
        attribute obj_to_be_cut, you will add obj_to_be_cut_props.
        - The value of this _props key will be a new dictionary containing relevant semantic property-value pairs. You must select 
        applicable properties from the Exhaustive Properties List provided below.
        - You should include as many relevant properties as you can reasonably infer or extract. Do not artificially limit the 
        number of pairs to one or two; the goal is to be as descriptive as possible.
        - Simplify the original attribute's value if you extract a descriptive adjective from it. For example, if the original 
        value was "black knife" and you add "color": "black" to the utensil_props, the original utensil value should be simplified to "knife".
    
    ---
    
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
    
    Output Rules:
    - Your response MUST be a single, updated, and valid JSON object.
    - Your response must NOT contain any other text, explanations, or markdown formatting. Only the raw JSON.
    - For attributes that are not relevant or are null, do not add a _props key for them.
    
    ### Examples ###
    Here are examples of how you must perform this task.
    
    Example 1:
    
    Input JSON:
    JSON
    
    
      "action_verb": "cut",
      "amount": null,
      "obj_to_be_cut": "apple",
      "unit": null,
      "utensil": "black knife"
    
    Your Output:
    JSON
    
    
      "action_verb": "cut",
      "amount": null,
      "obj_to_be_cut": "apple",
      "obj_to_be_cut_props": 
        "color": "red",
        "texture": "smooth",
        "shape": "round"
      ,
      "unit": null,
      "utensil": "knife",
      "utensil_props": 
        "color": "black",
        "edge": "sharp"
      
    ---
    
    In addition to action_roles, you're also prompted with some optional context, use the information from the context to get semantic
    knowledge for the entities if needed. Context if given should only be used as an add on knowledge. 
    
    Now perform the task for the given, attribute-value action_roles JSON
    
    action_roles : {action_roles} \n
    
    context : {context}
    
    /nothink
"""

cram_plan_prompt_template = """
    You are an expert CRAM Plan Generator. Your sole purpose is to take a generic CRAM plan syntax (a template) and an 
    enriched JSON data object, and generate a final, specific, and valid CRAM plan.
    
    Your Mission:

    You will receive two inputs:
    
    CRAM Plan Syntax: A template string for a CRAM plan with placeholders like key and key_props.
    Enriched JSON Data: A JSON object containing the values and semantic properties for the action.
    Your task is to meticulously populate the syntax template with the data from the JSON object, following three critical rules to produce the final plan.
    
    Execution Rules:
    1. Direct Substitution:
        - Replace all simple placeholders (e.g., obj_to_be_cut, utensil) with their corresponding string values from the JSON.
    2. Property Dictionary Formatting:
        - For property placeholders (e.g., obj_to_be_cut_props), you must take the corresponding _props JSON dictionary.
        - Convert this dictionary into a series of space-separated Lisp-style expressions. For example, 
        the JSON "color": "red", "edge": "sharp" becomes the string (color red) (edge sharp).
        - Insert this formatted string into the template. Note that there are no surrounding parentheses for the block itself, only for each individual pair.
    3. Strict Null Value Handling (Most Important Rule):
        - Before substitution, check if any placeholder corresponds to a null value in the JSON.
        - If a placeholder's value is null, you must find the immediate parent S-expression (the clause in parentheses ()) that 
        contains it and remove that entire clause from the final output.
        - For example, if amount is null in the JSON, the entire (count ...) clause in the syntax must be completely omitted from the generated plan.
        
    Output Format:
    - Your output MUST be the single, final CRAM plan string.
    - Your response must NOT contain any other text, explanations, or markdown formatting. Only the plain CRAM plan string.
    
    ### Examples ###
    Example 1: Cutting Action (with nulls)
    
    Input CRAM Plan Syntax:
    (perform (an action (type cut-object) (an object (type obj_to_be_cut) obj_to_be_cut_props )(count (unit unit)
    unit_props (number amount) amount_props)(utensil (an object (type utensil) utensil_props))))
    
    Input Enriched JSON Data:
    JSON
    
    
      "obj_to_be_cut": "apple",
      "obj_to_be_cut_props": "color": "red", "shape": "round",
      "amount": null,
      "unit": null,
      "utensil": "knife",
      "utensil_props": "material": "steel", "edge": "sharp"
    
    Your Output:
    (perform (an action (type cut-object) (an object (type apple) (color red) (shape round)) (utensil (an object (type knife) (material steel) (edge sharp)))))
    (Notice the (count ...) clause is completely gone because amount and unit were null.)
    
    Now perform the task for the given CRAM Plan Syntax and Enriched JSON Data:
    
    cram_plan_syntax : {cram_plan_syntax}
    enriched_json_data : {enriched_json_data}
    
    /nothink
"""

# Nodes
def action_node(state: StateModel):
    print("INSIDE ACTION NODE")

    instruction = state["instruction"]

    mod1_prompt = ChatPromptTemplate.from_template(mod1_template)

    chain = mod1_prompt | ollama_llm.with_structured_output(ADAction, method="json_schema")

    response = chain.invoke({'instruction': instruction})

    print(response)

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

def cram_node(state : StateModel):
    print("INSIDE UNIVERSAL NODE")
    instruction = state['instruction']
    action_core = state['action_core']
    required_fields = json_data[action_core]['action_roles']
    cram_plan = json_data[action_core]['cram_plan']
    # cram_plan = '\n'.join(cram_plan)
    print("cram_plan", cram_plan)
    print("-" * 10)

    action_cls = next(
        (cls for cls in action_classes if cls.__name__ == action_core or
         getattr(cls, 'action_core', None) == action_core),
        None
    )
    if action_cls is None:
        raise ValueError(f"Unknown action_core: {action_core}")

    field_prompt = ChatPromptTemplate.from_template(field_prompt_template)
    field_props_prompt = ChatPromptTemplate.from_template(field_props_prompt_template)
    cram_plan_prompt = ChatPromptTemplate.from_template(cram_plan_prompt_template)


    chain = field_prompt | ollama_llm
    response = chain.invoke({'instruction': instruction, 'action_core': action_core, 'target_attributes':required_fields})
    action_core_attributes = think_remover(response.content)
    print(action_core_attributes)
    print("-"*10)

    context = state.get('context', "")
    print("fCONTEXT", context)
    chain2 = field_props_prompt | ollama_llm
    response2 = chain2.invoke({'action_roles' : action_core_attributes, 'context' : context})
    enriched_action_core_attributes = think_remover(response2.content)
    print(enriched_action_core_attributes)
    print("-" * 10)

    chain3 = cram_plan_prompt | ollama_llm
    response3 = chain3.invoke({'cram_plan_syntax' : cram_plan, 'enriched_json_data' : enriched_action_core_attributes})
    cram_plan_response = think_remover(response3.content)
    print(cram_plan_response)
    print("-" * 10)

    # chain = uni_prompt | ollama_llm.with_structured_output(action_cls, method="json_schema")
    #
    # response = chain.invoke({'instruction': instruction, 'action_core': action_core, 'required_attributes': str(required_fields)})
    #
    # print(response)
    #
    # chain2 = cram_prompt | ollama_llm
    #
    # response2 = chain2.invoke({'action_core': response.model_dump_json(indent=2), 'syntax_cram_plan': cram_plan})
    #
    # clean_response = think_remover(response2.content)
    #
    # print("CRAM PLAN ", clean_response)

    return {'action_core_attributes': action_core_attributes, 'enriched_action_core_attributes': enriched_action_core_attributes,
            'cram_plan_response': cram_plan_response}


ad_builder = StateGraph(StateModel)
ad_builder.add_node("action_node", action_node)
ad_builder.add_node("cram_node", cram_node)
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
ad_builder.add_edge("action_node","cram_node")
ad_builder.add_edge("cram_node",END)

ad_graph = ad_builder.compile()

if __name__ == "__main__":
    print()

    # ress = ad_graph.invoke({"instruction": "pour the water from the mug into sink"})

    # print(ad_graph.invoke({"instruction": "pour 2 ounces of honey from the blue bottle onto the round white plate"}))
    #
    # print(ad_graph.invoke({"instruction": "cut the apple into 2 slices using black knife"}))

    print(ad_graph.invoke({"instruction": "pick up the cooking pan from the wooden drawer"}))

    # print(json_path)
    # print(json_data['Pouring'])
    # print(ress['designator'])