from src.langchain.llm_configuration import *
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import re
from typing import Dict
from pydantic import BaseModel, Field

reasoner_prompt_template = """
    You are a highly specialized information retrieval AI. Your sole function is to receive a large, structured context and a 
    user query. You must analyze the query, find the precisely relevant information within the context, and return it as a single, valid Python dictionary.

    ### Core Directives: ###
    
    Input: You will be provided with a CONTEXT block containing detailed, structured information (such as instructions, action cores, 
    attribute values, cram plans, Flanagan motion phases, and FrameNet data) and a QUERY.
    
    ### Processing: ###
    
    - Deconstruct the Query: First, meticulously analyze the QUERY to understand exactly what information is being requested.
    
    - Targeted Retrieval: Search the CONTEXT to locate the specific segment(s) that directly and completely answer the QUERY. 
    You must ignore all other information in the context.
    
    - Structure the Answer: Organize the retrieved information into a Python dictionary. If the data is hierarchical, represent it 
    using nested dictionaries. The keys and values of the dictionary should accurately reflect the information from the context.
    
    - Verification: Before outputting, verify that the dictionary you have constructed is a precise and accurate answer to the user's 
    QUERY and is derived exclusively from the provided CONTEXT.
    
    ### Output Constraint (Crucial): ###
    
    - Your response MUST be a single, valid Python dictionary.
    
    - Your response should only contain the information already present in the provided CONTEXT. You should not alter anything.
    
    - DO NOT include any introductory text, closing remarks, explanations, apologies, or any form of conversational filler.
    
    - Your entire response must start and end with curly braces like a dictionary.
    
    ## Example Interaction ##
    GIVEN:
    
    CONTEXT:
    
    
      "instruction_id": "INS-4815",
      "action_core": 
        "name": "PLACE",
        "theme": "the red block",
        "goal": 
          "type": "location_relation",
          "relation": "on_top_of",
          "anchor": "the blue cube"
        
      ,
      "cram_plan": 
        "plan_name": "place-object-on-another",
        "phases": [
          
            "phase_name": "reaching",
            "motion": "move-arm-to-object"
          ,
          
            "phase_name": "grasping",
            "motion": "close-gripper"
          
        ]
      ,
      "framenet_info": 
        "frame": "Placing",
        "lexical_units": ["place", "put", "set"]
      
    
    
    QUERY:
    
    What is the cram plan information?
    
    REQUIRED RESPONSE FROM YOU:
    
    
      "plan_name": "place-object-on-another",
      "phases": [
        
          "phase_name": "reaching",
          "motion": "move-arm-to-object"
        ,
        
          "phase_name": "grasping",
          "motion": "close-gripper"
        
      ]
    
    ---
    NOTE: Take the above example just as a reference
    
    Now, perform the task for the given context and user query,
    
    CONTEXT:
    
    {context}
    
    QUERY:
    
    {query}
    
    OUTPUT: 
    
"""

reasoner_prompt_template2 = """
    You are a headless, specialized data extraction and synthesis engine. Your function is to interpret a user's QUERY, locate all semantically 
    relevant information within a structured CONTEXT, and return a single, precise JSON object that completely answers the query.

    ### Core Task ###
    Interpret the semantic intent of the QUERY. Then, extract and synthesize all required data from the CONTEXT into a single, coherent, and complete JSON object.
    
    ### Operational Workflow ###
    - Interpret Query Intent: 
        - Analyze the QUERY to understand its underlying goal. Do not simply search for keywords. Determine what kind of information 
        constitutes a full answer. For example, a query about "how" to do something implies looking for execution steps, while a query about "what" is being done 
        implies looking for the core goal or theme.
    
    - Map Intent to Context & Extract: 
        - Scan the CONTEXT to understand the purpose of each major JSON object. Map the query's intent to the relevant object(s). Extract all
        key-value pairs, objects, and arrays necessary to fulfill that intent. This may require extracting data from multiple, separate parts of the CONTEXT.
    
    - Synthesize a Complete Answer:
        - Assemble the extracted data into a single, logical JSON object. If data was pulled from multiple sources, you MUST combine it into a 
        coherent structure that directly and fully answers the user's conceptual query. The final object should be self-contained and logical.
    
    - Format the Output: 
        Ensure the final output strictly adheres to the mandates below.
    
    ### Output Mandates (Non-negotiable) ###
    - Format:
        - The response MUST be a single, valid JSON object. The entire text output must start with curly brace (or [ if the root is an array) and end with curly brace (or ]).
    - Content Purity:
        - The output MUST ONLY contain data present in the provided CONTEXT. Do not infer, add, or alter any information. You may structure the 
            extracted data under new descriptive keys if it aids in synthesizing a coherent answer (as shown in Example 2).
    - Strict Exclusion:
        - There must be ABSOLUTELY NO introductory text, explanations, apologies, summaries, or any other conversational text in the response.
    
    ## Example Interactions ##
    Example 1: Direct Query
    GIVEN CONTEXT: 
    CONTEXT:
    
    
      "instruction_id": "INS-4815",
      "action_core": 
        "name": "PLACE",
        "theme": "the red block",
        "goal": 
          "type": "location_relation",
          "relation": "on_top_of",
          "anchor": "the blue cube"
        
      ,
      "cram_plan": 
        "plan_name": "place-object-on-another",
        "phases": [
          
            "phase_name": "reaching",
            "motion": "move-arm-to-object"
          ,
          
            "phase_name": "grasping",
            "motion": "close-gripper"
          
        ]
      ,
      "framenet_info": 
        "frame": "Placing",
        "lexical_units": ["place", "put", "set"]
      
    
    GIVEN QUERY: What is the cram plan information?
    REQUIRED RESPONSE:
    JSON
    
    {{
      "plan_name": "place-object-on-another",
      "phases": [
        {{
          "phase_name": "reaching",
          "motion": "move-arm-to-object"
        }},
        {{
          "phase_name": "grasping",
          "motion": "close-gripper"
        }}
      ]
    }}
    Example 2: Inferential / Synthesis Query
    GIVEN CONTEXT:
    
    JSON
    
    {{
      "instruction_id": "INS-4815",
      "action_core": {{
        "name": "PLACE",
        "theme": "the red block",
        "goal": {{
          "type": "location_relation",
          "relation": "on_top_of",
          "anchor": "the blue cube"
        }}
      }},
      "cram_plan": {{
        "plan_name": "place-object-on-another",
        "phases": [
          {{
            "phase_name": "reaching",
            "motion": "move-arm-to-object"
          }},
          {{
            "phase_name": "grasping",
            "motion": "close-gripper"
          }}
        ]
      }}
    }}
    
    GIVEN QUERY: Describe the complete action to be performed.
    REQUIRED RESPONSE: (This response synthesizes two different top-level keys into one logical answer)
    
    JSON
    
    {{
      "action_goal": {{
        "name": "PLACE",
        "theme": "the red block",
        "goal": {{
          "type": "location_relation",
          "relation": "on_top_of",
          "anchor": "the blue cube"
        }}
      }},
      "execution_plan": {{
        "plan_name": "place-object-on-another",
        "phases": [
          {{
            "phase_name": "reaching",
            "motion": "move-arm-to-object"
          }},
          {{
            "phase_name": "grasping",
            "motion": "close-gripper"
          }}
        ]
      }}
    }}
    
    ---
    NOTE: Take the above example just as a reference for patterns not for exact structures.
    
    Now, perform the task for the given context and user query,
    
    CONTEXT:
    
    {context}
    
    QUERY:
    
    {query}
    
    OUTPUT: 

"""

class JsonReasoner(BaseModel):
    content : Dict[str, str]

# json_parser = JsonOutputParser(pydantic_object=JsonReasoner)

def think_remover(res : str):
    if re.search(r"<think>.*?</think>", res, flags=re.DOTALL):
        cleaned_res = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip()
    else:
        cleaned_res = res.strip()

    return cleaned_res

reasoner_prompt2 = ChatPromptTemplate.from_template(reasoner_prompt_template2)

reasoner_chain = reasoner_prompt2 | ollama_llm.with_structured_output(JsonReasoner, method="json_schema")

def invoke_reasoner(context : str, query : str):

    print("Came to Reasoner")

    reasoner_response = reasoner_chain.invoke({'context': context, 'query': query})
    # clean_response = think_remover(reasoner_response['content'])

    print("Clean Reasoned Response ", reasoner_response)
    return reasoner_response.model_dump_json(indent=2)