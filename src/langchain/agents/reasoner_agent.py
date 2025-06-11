from src.langchain.llm_configuration import *
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import re

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


json_parser = JsonOutputParser()

def think_remover(res : str):
    if re.search(r"<think>.*?</think>", res, flags=re.DOTALL):
        cleaned_res = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip()
    else:
        cleaned_res = res.strip()

    return cleaned_res

reasoner_prompt = ChatPromptTemplate.from_template(reasoner_prompt_template)

reasoner_chain = reasoner_prompt | ollama_llm | json_parser

def invoke_reasoner(context : str, query : str):
    reasoner_response = reasoner_chain.invoke({'context': context, 'query': query})
    clean_response = think_remover(reasoner_response.content)
    return clean_response