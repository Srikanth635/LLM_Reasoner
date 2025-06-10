from src.langchain.llm_configuration import *
from pydantic import BaseModel, Field
from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate

class Segments(BaseModel):
    segments : List[Dict[str, str]] = Field(description="List of segmented information about an action")

class AtomicActions(BaseModel):
    atomic_actions : List[Dict[str, str]] = Field(description="List of atomic action information from a segmented action")

segment_prompt_template = """
    You are an intelligent code assistant and efficient text parser agent.

    Given input text that may contain a mix of normal text and JSON-formatted segments describing completed actions, your task is to extract information for each segment that includes:
    
    A natural language task description, with any leading segment label (e.g., "In Segment 1,") removed.
    
    The corresponding JSON-formatted action information.
    
    Input:
    Text that includes both plain text and one or more labeled segments. Each segment contains a natural language description of an action and a corresponding Generated JSON: block.
    
    Output:
    Return a list of dictionaries where each dictionary represents one segment and follows this format:
    "task_instruction": <natural_language_text>, "action_designator": <json_segment>
    Where:
    
    <natural_language_text> is the task description from the segment, with leading phrases like “In Segment 1,” removed.
    
    <json_segment> is the exact JSON content as provided.
    
    Ignore unrelated text outside of the segments. Do not provide any explanation or feedback—only output the list of extracted segment dictionaries in the format described.
    
    Given, 
    
    Input : {input_text}
    
"""

segment_prompt = ChatPromptTemplate.from_template(segment_prompt_template)

atomic_prompt_template = """
    You are a skilled semantic parsing agent specialized in robotic action modeling.

    Your task is to process the output from a prior system that contains:
    
    A natural language task_instruction.
    
    A combined JSON action_designator that includes multiple atomic actions (e.g., both picking_up and placing actions within a single JSON).
    
    Objective:
    Decompose the input into a list of atomic action designators, each representing one distinct action.
    For each, return:
    
    A task_instruction that is specific to that action, with resolved references (e.g., replace “it” with “the brown onion”).
    
    An action_designator that contains:
    
    All relevant shared context (e.g., component_information, grasp_descriptor, etc.)
    
    Only the individual action (primary_action or secondary_action) moved into a new executed_action.
    
    Input Format:

      "task_instruction": "<compound instruction>",
      "action_designator": "<stringified JSON with multiple actions>"

    Output Format:
    A list of dictionaries, where each dictionary has:

      "task_instruction": "<atomic task with explicit reference>",
      "action_designator": <JSON object containing only that atomic action>
    
    Processing Rules:
    Identify primary_action and secondary_action inside executed_action.
    
    For each:
    
    Create a new JSON object.
    
    Retain all fields from the original (e.g., component_id, component_information, etc.).
    
    Replace executed_action with a new dictionary that contains only the single action being represented.
    
    Resolve pronouns like “it” using earlier noun phrases (e.g., "a brown onion").
    
    Output must be valid Python dictionaries with valid embedded JSON objects (not empty strings).
    
    Example Input:

      "task_instruction": "the person picks up a brown onion and places it on a wooden cutting board",
      "action_designator": "...full JSON with both primary_action and secondary_action..."
    
    Expected Output:

    [
      
        "task_instruction": "the person picks up a brown onion",
        "action_designator": 
          "component_id": "...",
          "component_information": ..,
          "executed_action": 
            "primary_action": 
              "action_name": "picking_up",
              "hand": "right_hand"
            
          ,
          "grasp_descriptor": ...,
          "environmental_factors": ...
        
      ,
      
        "task_instruction": "the person places the brown onion on a wooden cutting board",
        "action_designator": 
          "component_id": "...",
          "component_information": ...,
          "executed_action": 
            "secondary_action": 
              "action_name": "placing",
              "hand": "right_hand"
            
          ,
          "grasp_descriptor": ...,
          "environmental_factors": ...
        
      
    ]
    Do not return empty action_designator fields. Only return the resulting list of parsed and properly structured atomic action designators.
    
    Given, input with combined atomic action descriptions
    
    input : {combined_action}
    
"""

atomic_prompt = ChatPromptTemplate.from_template(atomic_prompt_template)

chain1 = segment_prompt | ollama_llm.with_structured_output(Segments, method="json_schema")

chain2 = atomic_prompt | ollama_llm.with_structured_output(AtomicActions, method="json_schema")

if __name__ == "__main__":
    print()

    text = """
        Response:
        In Segment 1, the person picks up a brown onion and places it on a wooden cutting board.  
        In Segment 2, the person holds the knife and begins cutting the brown onion on the wooden cutting board.  
        In Segment 3, the person continues cutting the brown onion with the knife on the wooden cutting board.  
        In Segment 4, the person is still cutting the brown onion with the knife on the wooden cutting board.  
        In Segment 5, the person keeps cutting the brown onion with the knife on the wooden cutting board.  
        In Segment 6, the person continues cutting the brown onion with the knife on the wooden cutting board.  
        In Segment 7, the person is still cutting the brown onion with the knife on the wooden cutting board.  
        In Segment 8, the person switches to peeling the skin off the brown onion with the knife on the wooden cutting board.  
        In Segment 9, the person continues peeling the skin off the brown onion with the knife on the wooden cutting board.  
        In Segment 10, the person switches to peeling the skin off a white onion with their fingers on the wooden cutting board.  
        In Segment 11, the person continues peeling the skin off the white onion with their fingers on the wooden cutting board.
        Segment 1: Picking
        Segment 2: Holding
        Segment 3: Cutting
        Segment 4: Cutting
        Segment 5: Cutting
        Segment 6: Cutting
        Segment 7: Cutting
        Segment 8: Peeling
        Segment 9: Peeling
        Segment 10: Peeling
        Segment 11: Peeling
        
        === Segment 1 ===
        Original: In Segment 1, the person picks up a brown onion and places it on a wooden cutting board.  
        Generated JSON:
         {
          "component_id": "onion_01",
          "component_information": {
            "name": "onion",
            "id_number": "01",
            "component_type": "food_item",
            "shape": "spherical",
            "size": "medium",
            "handle": "no",
            "orientation": "upright",
            "weight": 1
          },
          "executed_action": {
            "primary_action": {
              "action_name": "picking_up",
              "hand": "right_hand"
            },
            "secondary_action": {
              "action_name": "placing",
              "hand": "right_hand"
            }
          },
          "grasp_descriptor": {
            "grasp_type": "spherical",
            "contact_points": "three_fingers",
            "holding_type": "one_handed",
            "hand_orientation": "top_to_bottom"
          },
          "environmental_factors": {
            "surface_conditions": "flat_surfaces"
          }
        }
        
        === Segment 2 ===
        Original: In Segment 2, the person holds the knife and begins cutting the brown onion on the wooden cutting board.  
        Generated JSON:
         {
          "component_id": "knife_01",
          "component_information": {
            "name": "knife",
            "id_number": "01",
            "component_type": "kitchen_object",
            "shape": "flat",
            "size": "medium",
            "handle": "yes",
            "orientation": "upright",
            "weight": 1
          },
          "executed_action": {
            "primary_action": {
              "action_name": "cutting",
              "hand": "right_hand"
            },
            "secondary_action": {
              "action_name": "holding",
              "hand": "left_hand"
            }
          },
          "grasp_descriptor": {
            "grasp_type": "flat",
            "contact_points": "three_fingers",
            "holding_type": "one_handed",
            "hand_orientation": "top_to_bottom"
          },
          "environmental_factors": {
            "surface_conditions": "flat_surfaces"
          }
        }
    """

    seg_instance = chain1.invoke({"input_text" : text})

    print(seg_instance.segments[0])
    print("-"*10)
    print(seg_instance.segments[1])
    print("-" * 10)

    atomic_instance = chain2.invoke({"combined_action" : seg_instance.segments[0]})

    print(atomic_instance.atomic_actions[0])
    print("-" * 10)
    print(atomic_instance.atomic_actions[1])
    print("-" * 10)

