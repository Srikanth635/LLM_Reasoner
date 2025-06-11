from src.langchain.llm_configuration import *
from pydantic import BaseModel, Field
from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
import asyncio

class Segments(BaseModel):
    segments : List[Dict[str, str]] = Field(description="List of segmented information about an action")

class AtomicActions(BaseModel):
    atomic_actions : List[Dict[str, str]] = Field(description="List of atomic actions information from a combined segmented action")

segment_prompt_template = """
    ### ROLE ###
    You are an automated text-to-JSON parsing engine.
    
    ### OBJECTIVE ###
    To process input text containing multiple segments and extract a natural language instruction and a corresponding JSON object from each.
    
    ### INSTRUCTIONS ###
    - Your input will contain sections of text that are not relevant. Ignore all text outside of blocks that start with `=== Segment`.
    - A segment is a block of text that begins with a `=== Segment ... ===` delimiter.
    - Within each segment, you will find two key pieces of information:
        1.  The task description, located on the line starting with `Original:`.
        2.  The action designator, which is the JSON object following the line `Generated JSON:`.
    
    ### PARSING RULES ###
    1.  For the `task_instruction` value: Use the text from the `Original:` line but remove the initial phrase (e.g., "In Segment 1, ").
    2.  For the `action_designator` value: Use the entire JSON object exactly as it appears. Do not alter its formatting or content.
    
    ### OUTPUT FORMAT ###
    - You must return a single JSON list `[]` containing one dictionary for each segment.
    - Each dictionary must have exactly two keys: `task_instruction` and `action_designator`.
    - Do not add any commentary, explanations, or other text before or after the JSON list.
    
    ### START PROCESSING ###
    Input Text:
    {input_text}
    
    /nothink
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

action_decomposer_prompt_template = """
    You are an advanced AI agent specializing in Natural Language Understanding and resilient action decomposition. Your primary 
    function is to parse a natural language task_instruction, identify all sequential atomic actions, and decompose the segment into a 
    corresponding list of valid JSON objects.
    
    ### CRITICAL CONSTRAINTS: ###
    - NEVER FAIL THE OUTPUT: 
        Under no circumstances should you ever output error messages, apologies, or conversational text like "errors in the provided..." inside the final JSON values. 
        Your sole job is to construct a valid list of JSON dictionaries.
    - USE ONLY PROVIDED DATA: 
        You must construct the action_designator for each new atomic action by intelligently reusing the information available in the 
        original segment's action_designator. Do not invent new properties.
    - MAINTAIN CONTEXT: 
        The component_id, component_information, and other relevant context from the original segment must be carried over to all the new atomic segments you create.
    
    ### Core Logic and Ambiguity Resolution: ###
    - THE SENTENCE IS THE PRIMARY GUIDE: 
        The definitive sequence of events comes from the task_instruction sentence. Identify the actions in the order they are described grammatically, 
        regardless of how they are labeled (primary or secondary) in the input JSON.
    - MAP ACTIONS: 
        For each action identified in the sentence (e.g., "holding," then "cutting"), find the corresponding action data within the entire original executed_action object.
    - GENERATE ATOMIC SEGMENTS: 
        Create one new atomic segment for each action identified in the sentence. The number of dictionaries in your output list must match the number of actions.
    
    ### Transformation Rules for Decomposition: ###
    For each atomic action you identify in the sentence's sequence:
    
    - Create an Atomic task_instruction: Write a new, clean sentence describing only that single action (e.g., simplify "begins cutting" to "cuts").
    - Create a Corresponding action_designator:
        - Copy the relevant parent context (component_id, component_information, environmental_factors).
        - Create a new executed_action object where the primary_action is the current atomic action you are processing.
        - Intelligently include other details: A grasp_descriptor is relevant for picking_up, holding, or using an object, but should be excluded for placing or releasing it.
            
    ### Output Format: ###
    Your final output must always be a JSON list of dictionaries. Do not include any other text, reasoning, or explanation.
    
    --- EXAMPLE 1: SINGLE ATOMIC ACTION ---
    Input:
    
    JSON
    
    
      "task_instruction": "the person is cutting the brown onion with the knife on the wooden cutting board.",
      "action_designator": 
        "component_id": "knife_01",
        "executed_action":  "primary_action":  "action_name": "cutting"  
      
    
    Output:
    
    JSON
    
    [
      
        "task_instruction": "the person is cutting the brown onion with the knife on the wooden cutting board.",
        "action_designator": 
          "component_id": "knife_01",
          "executed_action":  "primary_action":  "action_name": "cutting"  
        
      
    ]
    --- EXAMPLE 2: MULTIPLE (3) ATOMIC ACTIONS ---
    Input:
    
    JSON
    
    
      "task_instruction": "a person picks up the knife, carefully slices the onion, and then places the knife back on the counter.",
      "action_designator": 
        "component_id": "knife_01",
        "component_information":  "name": "knife", "size": "medium" ,
        "grasp_descriptor":  "grasp_type": "cylindrical", "holding_type": "one_handed" ,
        "environmental_factors":  "surface_conditions": "flat_surfaces" 
      
    
    Output:
    
    JSON
    
    [
      
        "task_instruction": "a person picks up the knife.",
        "action_designator": 
          "component_id": "knife_01",
          "component_information":  "name": "knife", "size": "medium" ,
          "executed_action": 
            "primary_action":  "action_name": "picking_up" 
          ,
          "grasp_descriptor":  "grasp_type": "cylindrical", "holding_type": "one_handed" ,
          "environmental_factors":  "surface_conditions": "flat_surfaces" 
        
      ,
      
        "task_instruction": "a person carefully slices the onion with the knife.",
        "action_designator": 
          "component_id": "knife_01",
          "component_information":  "name": "knife", "size": "medium" ,
          "executed_action": 
            "primary_action":  "action_name": "slicing" 
          ,
          "grasp_descriptor": "grasp_type": "cylindrical", "holding_type": "one_handed" ,
          "environmental_factors":  "surface_conditions": "flat_surfaces" 
        
      ,
      
        "task_instruction": "a person places the knife back on the counter.",
        "action_designator": 
          "component_id": "knife_01",
          "component_information":  "name": "knife", "size": "medium" ,
          "executed_action": 
            "primary_action":  "action_name": "placing" 
          ,
          "environmental_factors":  "surface_conditions": "flat_surfaces" 
        
      
    ]
    
    Now, perform the task on the given segment,
    
    segment : {segment}

"""

action_decomposer_prompt = ChatPromptTemplate.from_template(action_decomposer_prompt_template)

chain1 = segment_prompt | ollama_llm.with_structured_output(Segments, method="json_schema")

# chain2 = atomic_prompt | ollama_llm.with_structured_output(AtomicActions, method="json_schema")
chain2 = action_decomposer_prompt | ollama_llm.with_structured_output(AtomicActions, method="json_schema")


async def process_segment_async(segment_data):
    """
    An asynchronous worker function that processes a single segment.
    It assumes your 'chain2' object has an async method 'ainvoke'.
    """
    index, segment = segment_data
    # Use the asynchronous 'ainvoke' method
    response = await chain2.ainvoke({"segment": segment})
    return index + 1, {"segment": segment, "atomics": response.atomic_actions}

async def main():
    """
    The main asynchronous function to run all tasks concurrently.
    """
    # Create a list of tasks to run. We use enumerate to keep track of the original index.
    tasks = [process_segment_async(item) for item in enumerate(seg_instance.segments)]

    # Run all tasks concurrently and wait for them all to complete
    results = await asyncio.gather(*tasks)

    # Convert the list of (key, value) tuples back into a dictionary
    segment_atomics = dict(results)

    # Now you have your dictionary, same as before but created much faster
    print('-'*10)
    print(segment_atomics)
    print('-' * 10)




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
    half_text = """Response:
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
    
    === Segment 3 ===
    Original: In Segment 3, the person continues cutting the brown onion with the knife on the wooden cutting board.  
    Generated JSON:
     {
      "component_id": "onion_01",
      "component_information": {
        "name": "onion",
        "id_number": "01",
        "component_type": "food",
        "shape": "spherical",
        "size": "medium",
        "handle": "no",
        "orientation": "upright",
        "weight": 2
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
        "grasp_type": "spherical",
        "contact_points": "three_fingers",
        "holding_type": "one_handed",
        "hand_orientation": "top_to_bottom"
      },
      "environmental_factors": {
        "surface_conditions": "flat_surfaces"
      }
    }
    
    === Segment 4 ===
    Original: In Segment 4, the person is still cutting the brown onion with the knife on the wooden cutting board.  
    Generated JSON:
     {
      "component_id": "onion_01",
      "component_information": {
        "name": "onion",
        "id_number": "01",
        "component_type": "food",
        "shape": "spherical",
        "size": "medium",
        "handle": "no",
        "orientation": "upright",
        "weight": 2
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
        "grasp_type": "spherical",
        "contact_points": "three_fingers",
        "holding_type": "one_handed",
        "hand_orientation": "top_to_bottom"
      },
      "environmental_factors": {
        "surface_conditions": "flat_surfaces"
      }
    }
    
    === Segment 5 ===
    Original: In Segment 5, the person keeps cutting the brown onion with the knife on the wooden cutting board.  
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
        "orientation": "flat",
        "weight": 2
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
        "grasp_type": "spherical",
        "contact_points": "three_fingers",
        "holding_type": "one_handed",
        "hand_orientation": "top_to_bottom"
      },
      "environmental_factors": {
        "surface_conditions": "flat_surfaces"
      }
    }
    
    === Segment 6 ===
    Original: In Segment 6, the person continues cutting the brown onion with the knife on the wooden cutting board.  
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
        "weight": 2
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
        "grasp_type": "spherical",
        "contact_points": "three_fingers",
        "holding_type": "one_handed",
        "hand_orientation": "top_to_bottom"
      },
      "environmental_factors": {
        "surface_conditions": "flat_surfaces"
      }
    }
    
    === Segment 7 ===
    Original: In Segment 7, the person is still cutting the brown onion with the knife on the wooden cutting board.  
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
        "weight": 2
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
        "grasp_type": "spherical",
        "contact_points": "three_fingers",
        "holding_type": "one_handed",
        "hand_orientation": "top_to_bottom"
      },
      "environmental_factors": {
        "surface_conditions": "flat_surfaces"
      }
}
"""
    full_text = """
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
    
    === Segment 3 ===
    Original: In Segment 3, the person continues cutting the brown onion with the knife on the wooden cutting board.  
    Generated JSON:
     {
      "component_id": "onion_01",
      "component_information": {
        "name": "onion",
        "id_number": "01",
        "component_type": "food",
        "shape": "spherical",
        "size": "medium",
        "handle": "no",
        "orientation": "upright",
        "weight": 2
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
        "grasp_type": "spherical",
        "contact_points": "three_fingers",
        "holding_type": "one_handed",
        "hand_orientation": "top_to_bottom"
      },
      "environmental_factors": {
        "surface_conditions": "flat_surfaces"
      }
    }
    
    === Segment 4 ===
    Original: In Segment 4, the person is still cutting the brown onion with the knife on the wooden cutting board.  
    Generated JSON:
     {
      "component_id": "onion_01",
      "component_information": {
        "name": "onion",
        "id_number": "01",
        "component_type": "food",
        "shape": "spherical",
        "size": "medium",
        "handle": "no",
        "orientation": "upright",
        "weight": 2
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
        "grasp_type": "spherical",
        "contact_points": "three_fingers",
        "holding_type": "one_handed",
        "hand_orientation": "top_to_bottom"
      },
      "environmental_factors": {
        "surface_conditions": "flat_surfaces"
      }
    }
    
    === Segment 5 ===
    Original: In Segment 5, the person keeps cutting the brown onion with the knife on the wooden cutting board.  
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
        "orientation": "flat",
        "weight": 2
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
        "grasp_type": "spherical",
        "contact_points": "three_fingers",
        "holding_type": "one_handed",
        "hand_orientation": "top_to_bottom"
      },
      "environmental_factors": {
        "surface_conditions": "flat_surfaces"
      }
    }
    
    === Segment 6 ===
    Original: In Segment 6, the person continues cutting the brown onion with the knife on the wooden cutting board.  
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
        "weight": 2
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
        "grasp_type": "spherical",
        "contact_points": "three_fingers",
        "holding_type": "one_handed",
        "hand_orientation": "top_to_bottom"
      },
      "environmental_factors": {
        "surface_conditions": "flat_surfaces"
      }
    }
    
    === Segment 7 ===
    Original: In Segment 7, the person is still cutting the brown onion with the knife on the wooden cutting board.  
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
        "weight": 2
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
        "grasp_type": "spherical",
        "contact_points": "three_fingers",
        "holding_type": "one_handed",
        "hand_orientation": "top_to_bottom"
      },
      "environmental_factors": {
        "surface_conditions": "flat_surfaces"
      }
    }
    
    === Segment 8 ===
    Original: In Segment 8, the person switches to peeling the skin off the brown onion with the knife on the wooden cutting board.  
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
          "action_name": "peeling",
          "hand": "right_hand"
        },
        "secondary_action": {
          "action_name": "holding",
          "hand": "left_hand"
        }
      },
      "grasp_descriptor": {
        "grasp_type": "spherical",
        "contact_points": "fingers",
        "holding_type": "one_handed",
        "hand_orientation": "top_to_bottom"
      },
      "environmental_factors": {
        "surface_conditions": "flat_surfaces"
      }
    }
    
    === Segment 9 ===
    Original: In Segment 9, the person continues peeling the skin off the brown onion with the knife on the wooden cutting board.  
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
          "action_name": "peeling",
          "hand": "right_hand"
        },
        "secondary_action": {
          "action_name": "holding",
          "hand": "left_hand"
        }
      },
      "grasp_descriptor": {
        "grasp_type": "spherical",
        "contact_points": "fingers_and_palms",
        "holding_type": "one_handed",
        "hand_orientation": "top_to_bottom"
      },
      "environmental_factors": {
        "surface_conditions": "flat_surfaces"
      }
    }
    
    === Segment 10 ===
    Original: In Segment 10, the person switches to peeling the skin off a white onion with their fingers on the wooden cutting board.  
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
          "action_name": "peeling",
          "hand": "right_hand"
        },
        "secondary_action": {
          "action_name": "holding",
          "hand": "left_hand"
        }
      },
      "grasp_descriptor": {
        "grasp_type": "spherical",
        "contact_points": "fingers",
        "holding_type": "one_handed",
        "hand_orientation": "top_to_bottom"
      },
      "environmental_factors": {
        "surface_conditions": "flat_surfaces"
      }
    }
    
    === Segment 11 ===
    Original: In Segment 11, the person continues peeling the skin off the white onion with their fingers on the wooden cutting board.
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
          "action_name": "peeling",
          "hand": "right_hand"
        },
        "secondary_action": {
          "action_name": "holding",
          "hand": "left_hand"
        }
      },
      "grasp_descriptor": {
        "grasp_type": "spherical",
        "contact_points": "fingers",
        "holding_type": "one_handed",
        "hand_orientation": "top_to_bottom"
      },
      "environmental_factors": {
        "surface_conditions": "flat_surfaces"
      }
    }
    """

    seg_instance = chain1.invoke({"input_text" : text})

    print("No.of segments ", len(seg_instance.segments))
    print(seg_instance.segments[0])
    print("-"*10)
    print(seg_instance.segments[1])
    print("-" * 10)

    segment_atomics = {}

    ### FOR ASYNC OPERATION
    # if hasattr(seg_instance, 'segments') and len(seg_instance.segments) > 0:
    #     asyncio.run(main())


    ### FOR NORMAL SYNC OPERATION
    if len(seg_instance.segments) > 0:
        for ind, seg in enumerate(seg_instance.segments):
            response2 = chain2.invoke({"segment": seg})
            segment_atomics[ind+1] = {"segment" : seg, "atomics" : response2.atomic_actions}
            # print(response2)
            # print("-"*10)

    print(segment_atomics)
    # chain2 = action_decomposer_prompt | ollama_llm.with_structured_output(AtomicActions, method="json_schema")
    # response2 = chain2.invoke({"segment" : seg_instance.segments[1]})
    # print(response2)

    # atomic_instance = chain2.invoke({"combined_action" : seg_instance.segments[0]})
    #
    # print(atomic_instance.atomic_actions[0])
    # print("-" * 10)
    # print(atomic_instance.atomic_actions[1])
    # print("-" * 10)

